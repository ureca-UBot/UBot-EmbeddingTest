package com.ubot.thresholdlab;

import java.io.IOException;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVPrinter;
import org.apache.commons.csv.CSVRecord;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.core.env.Environment;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.test.context.TestPropertySource;

import com.ubot.embedding.service.EmbeddingProfileService;
import com.ubot.embedding.service.EmbeddingService;
import com.ubot.faq.dto.response.FaqSearchResponseDto;
import com.ubot.faq.retrieval.FaqSentenceFilterService;
import com.ubot.faq.service.FaqVectorService;

import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.Logger;
import ch.qos.logback.classic.spi.ILoggingEvent;
import ch.qos.logback.core.read.ListAppender;

/**
 * 임계값 결정 실험의 점수 수집기입니다. 판정은 하지 않고 점수만 남깁니다.
 *
 * <p>질문마다 다음을 CSV로 저장합니다.
 * <ul>
 *   <li>candidates.csv: 질문 벡터 검색 상위 {@value #CANDIDATE_K}개 FAQ와 질문 벡터 유사도(운영과 같은 검색)</li>
 *   <li>sentences.csv: 그 FAQ들의 답변 문장 각각의 유사도(운영 FaqSentenceFilterService가 계산하는 값 그대로)</li>
 *   <li>faq_catalog.csv: 후보에 나온 FAQ의 질문·답변 (라벨링 참고용)</li>
 *   <li>meta.txt: 환경·스냅샷 정보. 수집 전후 FAQ 변경 여부 확인용</li>
 * </ul>
 *
 * <p>임계값 두 개를 모두 열어 둔 채 한 번만 수집하고, 이후 임계값 조합은 분석 스크립트가 계산합니다.
 * 문장 점수는 운영 코드({@link FaqSentenceFilterService})를 임계값 -1로 호출하고, 그 클래스가 남기는
 * debug 로그의 인자를 가로채서 얻습니다. 문장 나누기·목록 처리·벡터 계산을 복사하지 않으므로 운영 로직과 어긋나지 않습니다.
 *
 * <p>DB에는 쓰지 않습니다(임베딩 프로필이 없을 때만 getCurrentProfileId가 한 행을 만듭니다).
 * 그래도 앱 전체가 뜨면서 스케줄러가 돌 수 있으니 공유 DB가 아니라 <b>스냅샷을 복원한 로컬 DB</b>로 실행하세요.
 *
 * <p>실행 (PowerShell):
 * <pre>
 *   $env:RUN_THRESHOLD_LAB = "true"
 *   $env:THRESHOLD_LAB_QUESTIONS = "threshold-lab/questions.csv"   # 기본값
 *   $env:THRESHOLD_LAB_OUT = "build/threshold-lab"                 # 기본값
 *   .\gradlew.bat test --tests "com.ubot.thresholdlab.ThresholdScoreCollector" -i --rerun
 * </pre>
 *
 * <p>입력 CSV 머리글: id,group,expression_type,question,writer (id와 question만 사용)
 */
@SpringBootTest
@TestPropertySource(properties = {
        // 공유 DB에 마이그레이션이 적용되지 않게 하고, 답변 Worker가 큐를 읽지 않게 합니다.
        "spring.flyway.enabled=false",
        "app.chat.worker.polling-enabled=false"
})
@EnabledIfEnvironmentVariable(named = "RUN_THRESHOLD_LAB", matches = "true")
class ThresholdScoreCollector {

    /** 라벨링과 분석에서 후보를 넉넉히 보려고 운영의 Top-K(3)보다 크게 모읍니다. 운영 모의는 분석 스크립트가 상위 3개로 합니다. */
    static final int CANDIDATE_K = 10;

    @Autowired FaqVectorService faqVectorService;
    @Autowired EmbeddingService embeddingService;
    @Autowired EmbeddingProfileService embeddingProfileService;
    @Autowired JdbcTemplate jdbcTemplate;
    @Autowired Environment environment;

    @Test
    void collect() throws Exception {
        Path questionsPath = Path.of(env("THRESHOLD_LAB_QUESTIONS", "threshold-lab/questions.csv"));
        Path outDir = Path.of(env("THRESHOLD_LAB_OUT", "build/threshold-lab"));
        Files.createDirectories(outDir);

        List<Question> questions = readQuestions(questionsPath);
        if (questions.isEmpty()) {
            throw new IllegalStateException("질문이 없습니다: " + questionsPath.toAbsolutePath());
        }

        LocalDateTime startedAt = LocalDateTime.now();
        Snapshot before = snapshot();

        // 문장 점수는 운영 클래스를 임계값 -1(전부 통과)로 호출하고, 그 클래스의 debug 로그 인자를 가로채서 얻습니다.
        FaqSentenceFilterService scorer = new FaqSentenceFilterService(embeddingService, -1.0);
        Logger filterLogger = (Logger) LoggerFactory.getLogger(FaqSentenceFilterService.class);
        Level previousLevel = filterLogger.getLevel();
        boolean previousAdditive = filterLogger.isAdditive();
        ListAppender<ILoggingEvent> captured = new ListAppender<>();
        captured.start();
        filterLogger.addAppender(captured);
        filterLogger.setLevel(Level.DEBUG);
        filterLogger.setAdditive(false); // 문장마다 나오는 debug 로그가 콘솔을 채우지 않게 합니다.

        Map<Long, FaqSearchResponseDto> catalog = new LinkedHashMap<>();
        int succeeded = 0;
        List<String[]> errors = new ArrayList<>();

        try (CSVPrinter candidates = printer(outDir.resolve("candidates.csv"),
                "qid", "rank", "faq_id", "variant_id", "similarity", "intent", "faq_question", "variant_question");
                CSVPrinter sentences = printer(outDir.resolve("sentences.csv"),
                        "qid", "faq_id", "sentence_idx", "similarity", "sentence")) {

            int done = 0;
            for (Question question : questions) {
                done++;
                try {
                    FaqVectorService.SearchResult result =
                            faqVectorService.searchWithVector(question.text(), null, CANDIDATE_K);
                    List<FaqSearchResponseDto> faqs = result.faqs();

                    for (int index = 0; index < faqs.size(); index++) {
                        FaqSearchResponseDto faq = faqs.get(index);
                        catalog.putIfAbsent(faq.faqId(), faq);
                        candidates.printRecord(question.id(), index + 1, faq.faqId(), faq.variantId(),
                                faq.similarityScore(), faq.intent(), faq.question(), faq.variantQuestion());
                    }

                    captured.list.clear();
                    if (!faqs.isEmpty()) {
                        scorer.filter(result.queryVector(), faqs);
                    }
                    Map<Long, Integer> sentenceCounter = new HashMap<>();
                    for (ILoggingEvent event : new ArrayList<>(captured.list)) {
                        Object[] args = event.getArgumentArray();
                        // 운영 로그 인자 순서: FAQ ID, 유사도, 기준값, 결과, 문장
                        if (args == null || args.length < 5) {
                            continue;
                        }
                        Long faqId = (Long) args[0];
                        double similarity = ((Number) args[1]).doubleValue();
                        String sentence = String.valueOf(args[4]);
                        int sentenceIndex = sentenceCounter.merge(faqId, 1, Integer::sum) - 1;
                        sentences.printRecord(question.id(), faqId, sentenceIndex, similarity, sentence);
                    }
                    succeeded++;
                } catch (Exception exception) {
                    errors.add(new String[] {question.id(), String.valueOf(exception)});
                }
                if (done % 20 == 0 || done == questions.size()) {
                    System.out.printf("[threshold-lab] %d/%d 완료 (실패 %d)%n", done, questions.size(), errors.size());
                }
            }
        } finally {
            filterLogger.detachAppender(captured);
            filterLogger.setLevel(previousLevel);
            filterLogger.setAdditive(previousAdditive);
        }

        try (CSVPrinter catalogPrinter = printer(outDir.resolve("faq_catalog.csv"),
                "faq_id", "intent", "question", "answer")) {
            for (FaqSearchResponseDto faq : catalog.values()) {
                catalogPrinter.printRecord(faq.faqId(), faq.intent(), faq.question(), faq.answer());
            }
        }
        try (CSVPrinter errorPrinter = printer(outDir.resolve("errors.csv"), "qid", "error")) {
            for (String[] error : errors) {
                errorPrinter.printRecord((Object[]) error);
            }
        }

        Snapshot after = snapshot();
        writeMeta(outDir.resolve("meta.txt"), questionsPath, questions.size(), succeeded, errors.size(),
                startedAt, before, after);

        System.out.println("[threshold-lab] 저장 위치: " + outDir.toAbsolutePath());
        if (!before.equals(after)) {
            System.out.println("[threshold-lab] 경고: 수집 중 FAQ가 바뀌었습니다. meta.txt를 확인하고 다시 수집하세요.");
        }
        if (!errors.isEmpty()) {
            System.out.println("[threshold-lab] 경고: 실패한 질문이 있습니다. errors.csv를 확인하세요.");
        }
    }

    private record Question(String id, String text) {
    }

    private record Snapshot(long faqCount, long maxFaqId, long versionSum, long embeddingCount) {
    }

    private Snapshot snapshot() {
        Long profileId = embeddingProfileService.getCurrentProfileId();
        Map<String, Object> faq = jdbcTemplate.queryForMap(
                "SELECT COUNT(*) AS cnt, COALESCE(MAX(id), 0) AS max_id, COALESCE(SUM(version), 0) AS version_sum "
                        + "FROM faq WHERE deleted_at IS NULL");
        Long embeddings = jdbcTemplate.queryForObject(
                "SELECT COUNT(*) FROM faq_embeddings WHERE profile_id = ? AND vector_type = 'QUESTION'",
                Long.class, profileId);
        return new Snapshot(
                ((Number) faq.get("cnt")).longValue(),
                ((Number) faq.get("max_id")).longValue(),
                ((Number) faq.get("version_sum")).longValue(),
                embeddings == null ? 0 : embeddings);
    }

    private void writeMeta(Path path, Path questionsPath, int total, int succeeded, int failed,
            LocalDateTime startedAt, Snapshot before, Snapshot after) throws IOException {
        LocalDateTime finishedAt = LocalDateTime.now();
        List<String> lines = List.of(
                "시작=" + startedAt,
                "종료=" + finishedAt,
                "소요=" + Duration.between(startedAt, finishedAt).toSeconds() + "초",
                "질문파일=" + questionsPath.toAbsolutePath(),
                "질문수=" + total + ", 성공=" + succeeded + ", 실패=" + failed,
                "후보수(K)=" + CANDIDATE_K,
                "AI모드=" + environment.getProperty("app.ai.mode"),
                "임베딩엔진=" + environment.getProperty("app.ai.embedding-engine"),
                "임베딩모델=" + embeddingService.getModelName(),
                "임베딩프로필버전=" + environment.getProperty("app.ai.embedding-profile-version"),
                "임베딩프로필ID=" + embeddingProfileService.getCurrentProfileId(),
                "수집전 FAQ수=" + before.faqCount() + ", 최대ID=" + before.maxFaqId()
                        + ", 버전합=" + before.versionSum() + ", QUESTION벡터수=" + before.embeddingCount(),
                "수집후 FAQ수=" + after.faqCount() + ", 최대ID=" + after.maxFaqId()
                        + ", 버전합=" + after.versionSum() + ", QUESTION벡터수=" + after.embeddingCount(),
                "FAQ변경여부=" + (before.equals(after) ? "없음" : "있음(다시 수집 필요)"));
        Files.write(path, lines, StandardCharsets.UTF_8);
    }

    private List<Question> readQuestions(Path path) throws IOException {
        String content = Files.readString(path, StandardCharsets.UTF_8);
        if (content.startsWith("﻿")) {
            content = content.substring(1); // 엑셀이 붙이는 BOM
        }
        List<Question> questions = new ArrayList<>();
        try (CSVParser parser = CSVParser.parse(content, CSVFormat.DEFAULT.builder()
                .setHeader().setSkipHeaderRecord(true).setIgnoreEmptyLines(true).setTrim(true).build())) {
            for (CSVRecord record : parser) {
                String id = record.get("id");
                String text = record.get("question");
                if (id.isBlank() || text.isBlank()) {
                    continue;
                }
                questions.add(new Question(id, text));
            }
        }
        return questions;
    }

    private CSVPrinter printer(Path path, String... header) throws IOException {
        Writer writer = Files.newBufferedWriter(path, StandardCharsets.UTF_8);
        return new CSVPrinter(writer, CSVFormat.DEFAULT.builder().setHeader(header).build());
    }

    private static String env(String name, String defaultValue) {
        String value = System.getenv(name);
        return value == null || value.isBlank() ? defaultValue : value;
    }
}
