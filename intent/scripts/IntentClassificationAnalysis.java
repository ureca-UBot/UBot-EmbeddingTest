package com.ubot.embedding.analysis;

import com.pgvector.PGvector;
import com.ubot.PgvectorTestConfiguration;
import com.ubot.embedding.service.EmbeddingService;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVRecord;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.DisabledIfEnvironmentVariable;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.context.annotation.Import;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.context.TestPropertySource;

import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;

/**
 * 이 테스트의 목적: 사용자 질문을 Top-K(K=3)로 검색해서, 그중 threshold 이상인
 * 결과만 실제로 사용하는 구조에서 - "기존 FAQ 중 어느 것과도 유사하지 않은 질문을
 * 정확히 걸러내는 threshold"를 찾는 것이다.
 *
 * 검색 결과 각각은 독립적으로 자기 역할을 수행한다(일반=답변 반환,
 * 매장검색관련=매장 서비스 호출, 사용자검색관련=사용자 서비스 호출). 여러 개가
 * 동시에 살아남는 것 자체는 실패가 아니다 - 실제로 코퍼스에 그만큼 유사한 FAQ가
 * 있다면, 각각의 서비스가 호출되는 게 정상 동작이다. 따라서 "혼입"이나 "정답
 * 집합과의 완전 일치"는 이 테스트에서 측정하지 않는다.
 *
 * 측정하는 것은 딱 두 가지다:
 * - 그룹별 Recall: 실제로 코퍼스에 유사한 FAQ가 있는 질문에서, 검색이 그걸
 * 제대로 찾아내는가
 * - 무관 정확도: 실제로 코퍼스에 유사한 FAQ가 하나도 없는 질문에서, 검색
 * 결과가 정확히 비어있는가 (threshold를 넘는 게 하나도 없는가)
 *
 * 코퍼스 CSV 형식: (질문, 권장 처리 의도[FAQ_RAG/MAP_API/USER_INFO]) 2컬럼,
 * 팀원 원본 데이터를 그대로 사용한다 (불균형 그대로).
 * 테스트셋 CSV 형식: (질문, 의도) 2컬럼. "무관"은 정답이 없음(빈 집합)을 뜻한다.
 *
 * 실제 운영 테이블(faq)이나 FaqVectorRepository/FaqSearchResponseDto는 건드리지
 * 않고, 이 분석 전용 임시 테이블(faq_intent_analysis)에 코퍼스를 적재해서 직접
 * 검색한다.
 *
 * 실행 조건: 로컬/Docker Ollama 실행 중, Docker 실행 중.
 * application-test.yml은 수정하지 않고, @TestPropertySource로 이 테스트에서만
 * Ollama 주소를 재정의한다.
 */
@SpringBootTest
@Import(PgvectorTestConfiguration.class)
@ActiveProfiles("test")
@TestPropertySource(properties = {
        "ollama.base-url=http://localhost:11435",
        "spring.ai.ollama.base-url=http://localhost:11435"
})
@DisabledIfEnvironmentVariable(named = "CI", matches = "true")
class IntentClassificationAnalysis {

    private static final int K = 3;
    private static final int EMBED_BATCH_SIZE = 10;
    private static final double[] THRESHOLD_CANDIDATES = {
            0.600, 0.605, 0.610, 0.615, 0.620, 0.625, 0.630, 0.635, 0.640, 0.645,
            0.650, 0.655, 0.660, 0.665, 0.670, 0.675, 0.680, 0.685, 0.690, 0.695,
            0.700, 0.705, 0.710, 0.715, 0.720, 0.725, 0.730, 0.735, 0.740, 0.745,
            0.750, 0.755, 0.760, 0.765, 0.770, 0.775, 0.780, 0.785, 0.790, 0.795
    };
    private static final Set<String> ROUTABLE_GROUPS = Set.of("일반", "매장검색관련", "사용자검색관련");
    private static final Map<String, String> INTENT_TO_GROUP = Map.of(
            "FAQ_RAG", "일반",
            "MAP_API", "매장검색관련",
            "USER_INFO", "사용자검색관련");

    private static final String CORPUS_RESOURCE = "threshold/faq_intent_corpus.csv";
    private static final String TEST_SET_RESOURCE = "threshold/intent_test_set.csv";

    @Autowired
    private EmbeddingService embeddingService;

    @Autowired
    private JdbcTemplate jdbcTemplate;

    /** trueIntents가 비어있으면 "무관"(코퍼스에 유사한 FAQ가 없어야 함)을 뜻한다. */
    private record TestCase(Set<String> trueIntents, String question) {
    }

    private record CandidateHit(String group, String matchedQuestion, double similarity) {
    }

    @Test
    void intentClassificationAnalysis() throws IOException {
        printMetricExplanation();

        seedCorpus();
        List<TestCase> cases = loadTestCases();
        List<List<CandidateHit>> topKHits = evaluateTopK(cases);

        Path detailFile = writeCandidateDetailsToFile(cases, topKHits);
        Path summaryFile = writeSummaryReportToFile(cases, topKHits);
        Path recallFailureFile = writeRecallFailuresToFile(cases, topKHits);
        Path irrelevantFailureFile = writeIrrelevantFailuresToFile(cases, topKHits);

        System.out.println("\n상세 매칭 내역: " + detailFile.toAbsolutePath());
        System.out.println("요약 리포트: " + summaryFile.toAbsolutePath());
        System.out.println("재현 실패 목록: " + recallFailureFile.toAbsolutePath());
        System.out.println("무관 판정 실패 목록: " + irrelevantFailureFile.toAbsolutePath());
    }

    private void printMetricExplanation() {
        System.out.println("""

                ============================================================
                [지표 설명] "코퍼스와 유사하지 않은 질문을 걸러내는 threshold" 탐색

                각 테스트 질문의 정답은 "코퍼스에 실제로 유사한 FAQ가 있는 그룹의
                집합"이다(무관=빈 집합). Top-3 후보 중 threshold 이상인 것들의
                의도를 모아 "검색 결과 집합"을 만든다.

                여러 그룹이 동시에 살아남는 것 자체는 실패가 아니다 - 각 결과는
                독립적으로 자기 역할(답변 반환/매장 서비스 호출/사용자 서비스
                호출)을 수행하면 되기 때문이다. 그래서 "혼입"이나 "정답 집합과의
                완전 일치"는 측정하지 않는다.

                그룹별 Recall = 정답 집합에 그 의도가 포함된 질문들 중,
                               검색 결과에도 그 의도가 포함된 비율
                무관 정확도   = 정답이 무관(빈 집합)인 질문 중,
                               검색 결과도 정확히 빈 집합인 비율
                ============================================================
                """);
    }

    /** 팀원 원본 데이터를 그대로(질문, 의도) 2컬럼으로 읽어 적재한다. */
    private void seedCorpus() throws IOException {
        jdbcTemplate.execute("DROP TABLE IF EXISTS faq_intent_analysis");
        jdbcTemplate.execute("""
                CREATE TABLE faq_intent_analysis (
                    id BIGSERIAL PRIMARY KEY,
                    group_name TEXT NOT NULL,
                    question TEXT NOT NULL,
                    vector vector(1024)
                )
                """);

        List<CSVRecord> records = readCsv(CORPUS_RESOURCE);
        System.out.println("의도 분류 코퍼스 임베딩 시작: 총 " + records.size() + "개");

        embeddingService.embedText("워밍업");

        for (int i = 0; i < records.size(); i += EMBED_BATCH_SIZE) {
            List<CSVRecord> chunk = records.subList(i, Math.min(i + EMBED_BATCH_SIZE, records.size()));
            List<String> questions = chunk.stream().map(r -> r.get(0)).toList();
            List<PGvector> vectors = embeddingService.embedTexts(questions);

            for (int j = 0; j < chunk.size(); j++) {
                CSVRecord r = chunk.get(j);
                String group = INTENT_TO_GROUP.get(r.get(1));
                if (group == null) {
                    throw new IllegalStateException("알 수 없는 의도값: " + r.get(1));
                }
                jdbcTemplate.update(
                        "INSERT INTO faq_intent_analysis (group_name, question, vector) VALUES (?, ?, ?)",
                        group, r.get(0), vectors.get(j));
            }
            System.out.printf("  임베딩 진행: %d/%d%n", Math.min(i + EMBED_BATCH_SIZE, records.size()), records.size());
        }
        System.out.println("코퍼스 적재 완료. 총 " + records.size() + "개.");
    }

    /** "일반+매장검색관련"처럼 "+"로 구분된 의도를 집합으로 파싱한다. "무관"은 빈 집합. */
    private Set<String> parseIntents(String raw) {
        if (raw == null || raw.isBlank() || raw.equals("무관")) {
            return Set.of();
        }
        return new LinkedHashSet<>(Arrays.asList(raw.split("\\+")));
    }

    private List<TestCase> loadTestCases() throws IOException {
        return readCsv(TEST_SET_RESOURCE).stream()
                .map(r -> new TestCase(parseIntents(r.get(1)), r.get(0)))
                .toList();
    }

    private List<List<CandidateHit>> evaluateTopK(List<TestCase> cases) {
        List<List<CandidateHit>> results = new ArrayList<>();
        System.out.println("Top-" + K + " 통합 검색 시작: 총 " + cases.size() + "개");

        String sql = """
                SELECT group_name, question, 1 - (vector <=> ?) AS similarity_score
                FROM faq_intent_analysis
                ORDER BY vector <=> ?
                LIMIT ?
                """;

        for (int i = 0; i < cases.size(); i++) {
            TestCase c = cases.get(i);
            PGvector queryVector = embeddingService.embedText(c.question());

            List<CandidateHit> hits = jdbcTemplate.query(sql,
                    (rs, rowNum) -> new CandidateHit(
                            rs.getString("group_name"),
                            rs.getString("question"),
                            rs.getDouble("similarity_score")),
                    queryVector, queryVector, K);
            results.add(hits);

            if ((i + 1) % 50 == 0) {
                System.out.printf("  평가 진행: %d/%d%n", i + 1, cases.size());
            }
        }
        return results;
    }

    private Set<String> survivingIntents(List<CandidateHit> hits, double threshold) {
        Set<String> intents = new LinkedHashSet<>();
        for (CandidateHit hit : hits) {
            if (hit.similarity() >= threshold) {
                intents.add(hit.group());
            }
        }
        return intents;
    }

    private Path resolveOutputFile(String prefix) {
        String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss"));
        return Path.of("build/reports/intent-analysis/" + prefix + "-" + timestamp + ".txt");
    }

    /** 질문별 Top-K 매칭 상세(어떤 FAQ와 얼마나 비슷했는지)를 파일로 저장한다. */
    private Path writeCandidateDetailsToFile(List<TestCase> cases, List<List<CandidateHit>> topKHits)
            throws IOException {
        Path outputFile = resolveOutputFile("candidate-details");
        Files.createDirectories(outputFile.getParent());

        try (PrintWriter writer = new PrintWriter(Files.newBufferedWriter(outputFile, StandardCharsets.UTF_8))) {
            writer.println("=== 질문별 Top-" + K + " 매칭 상세 ===");
            for (int i = 0; i < cases.size(); i++) {
                TestCase c = cases.get(i);
                String trueLabel = c.trueIntents().isEmpty() ? "무관" : String.join("+", c.trueIntents());
                writer.printf("[%s] \"%s\"%n", trueLabel, c.question());
                for (CandidateHit hit : topKHits.get(i)) {
                    writer.printf("   -> [%s] %.6f | %s%n", hit.group(), hit.similarity(), hit.matchedQuestion());
                }
            }
        }
        return outputFile;
    }

    /** threshold별 요약 지표(그룹별 Recall, 무관 정확도)를 파일로 저장한다. */
    private Path writeSummaryReportToFile(List<TestCase> cases, List<List<CandidateHit>> topKHits) throws IOException {
        Path outputFile = resolveOutputFile("summary-report");
        Files.createDirectories(outputFile.getParent());

        try (PrintWriter writer = new PrintWriter(Files.newBufferedWriter(outputFile, StandardCharsets.UTF_8))) {
            for (double threshold : THRESHOLD_CANDIDATES) {
                writeSummaryForThreshold(writer, cases, topKHits, threshold);
            }
        }
        return outputFile;
    }

    private void writeSummaryForThreshold(PrintWriter writer, List<TestCase> cases,
            List<List<CandidateHit>> topKHits, double threshold) {
        Map<String, Integer> trueCountByGroup = new LinkedHashMap<>();
        Map<String, Integer> recalledCountByGroup = new LinkedHashMap<>();
        for (String g : ROUTABLE_GROUPS) {
            trueCountByGroup.put(g, 0);
            recalledCountByGroup.put(g, 0);
        }

        int muhwanTotal = 0;
        int muhwanCorrect = 0;

        for (int i = 0; i < cases.size(); i++) {
            TestCase c = cases.get(i);
            Set<String> predicted = survivingIntents(topKHits.get(i), threshold);
            Set<String> trueSet = c.trueIntents();

            for (String g : trueSet) {
                trueCountByGroup.merge(g, 1, Integer::sum);
                if (predicted.contains(g)) {
                    recalledCountByGroup.merge(g, 1, Integer::sum);
                }
            }

            if (trueSet.isEmpty()) {
                muhwanTotal++;
                if (predicted.isEmpty()) {
                    muhwanCorrect++;
                }
            }
        }

        writer.printf("%n=== threshold=%.6f ===%n", threshold);
        for (String g : ROUTABLE_GROUPS) {
            int total = trueCountByGroup.get(g);
            double recall = total == 0 ? 0.0 : (double) recalledCountByGroup.get(g) / total;
            writer.printf("  [%s] 재현율=%.6f (정답에 포함된 질문 %d개 중 %d개 회수)%n",
                    g, recall, total, recalledCountByGroup.get(g));
        }

        double muhwanAccuracy = muhwanTotal == 0 ? 0.0 : (double) muhwanCorrect / muhwanTotal;
        writer.printf("  [무관 정확도] %.6f (%d/%d건이 정확히 빈 결과로 판정됨)%n",
                muhwanAccuracy, muhwanCorrect, muhwanTotal);
    }

    /**
     * 놓친 의도 하나가 왜 놓쳤는지를 두 가지로 구분해서 문구로 만든다.
     * - Top-3 안에 그 의도 후보가 있었는데 유사도가 threshold 미만이라 탈락한 경우
     * → "threshold 미달"로 표시하고 실제 유사도를 함께 보여준다. threshold를
     * 낮추면 해결될 가능성이 있다.
     * - Top-3 안에 그 의도 후보가 애초에 하나도 없었던 경우
     * → "Top-3 후보 없음"으로 표시한다. threshold를 아무리 낮춰도 해결되지
     * 않으며 K값이나 코퍼스 구조를 봐야 한다.
     */
    private String describeMissedIntent(String missedIntent, List<CandidateHit> hits, double threshold) {
        double bestSimilarity = hits.stream()
                .filter(h -> h.group().equals(missedIntent))
                .mapToDouble(CandidateHit::similarity)
                .max()
                .orElse(Double.NaN);

        if (Double.isNaN(bestSimilarity)) {
            return missedIntent + "(Top-3 후보 없음)";
        }
        return String.format("%s(threshold 미달, 최고 유사도=%.6f < %.6f)",
                missedIntent, bestSimilarity, threshold);
    }

    /**
     * 재현에 실패한 케이스(정답 의도 중 일부가 threshold를 통과하지 못해 검색
     * 결과에서 빠진 경우)만 threshold별로 모아서 파일로 저장한다. 무관 케이스는
     * 대상이 아니므로 건너뛴다.
     */
    private Path writeRecallFailuresToFile(List<TestCase> cases, List<List<CandidateHit>> topKHits)
            throws IOException {
        Path outputFile = resolveOutputFile("recall-failures");
        Files.createDirectories(outputFile.getParent());

        try (PrintWriter writer = new PrintWriter(Files.newBufferedWriter(outputFile, StandardCharsets.UTF_8))) {
            writer.println("=== threshold별 재현 실패 케이스 (정답 의도인데 검색 결과에서 빠짐) ===");
            for (double threshold : THRESHOLD_CANDIDATES) {
                writer.printf("%n=== threshold=%.6f ===%n", threshold);
                boolean any = false;

                for (int i = 0; i < cases.size(); i++) {
                    TestCase c = cases.get(i);
                    if (c.trueIntents().isEmpty()) {
                        continue; // 무관 케이스는 재현 실패 대상이 아님
                    }

                    List<CandidateHit> hits = topKHits.get(i);
                    Set<String> predicted = survivingIntents(hits, threshold);
                    Set<String> missed = new LinkedHashSet<>(c.trueIntents());
                    missed.removeAll(predicted);

                    if (!missed.isEmpty()) {
                        any = true;
                        String missedLabel = missed.stream()
                                .map(intent -> describeMissedIntent(intent, hits, threshold))
                                .reduce((a, b) -> a + " + " + b)
                                .orElse("");
                        writer.printf("[놓친 의도: %s] \"%s\"%n", missedLabel, c.question());
                        for (CandidateHit hit : hits) {
                            writer.printf("   -> [%s] %.6f | %s%n",
                                    hit.group(), hit.similarity(), hit.matchedQuestion());
                        }
                    }
                }

                if (!any) {
                    writer.println("  (재현 실패 없음)");
                }
            }
        }
        return outputFile;
    }

    /**
     * 무관 판정에 실패한 케이스(정답이 무관인데 threshold를 통과한 후보가 하나
     * 이상 남아 검색 결과가 비지 않은 경우)만 threshold별로 모아서 파일로
     * 저장한다.
     */
    private Path writeIrrelevantFailuresToFile(List<TestCase> cases, List<List<CandidateHit>> topKHits)
            throws IOException {
        Path outputFile = resolveOutputFile("irrelevant-failures");
        Files.createDirectories(outputFile.getParent());

        try (PrintWriter writer = new PrintWriter(Files.newBufferedWriter(outputFile, StandardCharsets.UTF_8))) {
            writer.println("=== threshold별 무관 판정 실패 케이스 (정답은 무관인데 결과가 비지 않음) ===");
            for (double threshold : THRESHOLD_CANDIDATES) {
                writer.printf("%n=== threshold=%.6f ===%n", threshold);
                boolean any = false;

                for (int i = 0; i < cases.size(); i++) {
                    TestCase c = cases.get(i);
                    if (!c.trueIntents().isEmpty()) {
                        continue; // 정답이 있는 케이스는 무관 판정 실패 대상이 아님
                    }

                    Set<String> predicted = survivingIntents(topKHits.get(i), threshold);
                    if (!predicted.isEmpty()) {
                        any = true;
                        writer.printf("[정답: 무관 / 오탐으로 나온 의도: %s] \"%s\"%n",
                                String.join("+", predicted), c.question());
                        for (CandidateHit hit : topKHits.get(i)) {
                            writer.printf("   -> [%s] %.6f | %s%n",
                                    hit.group(), hit.similarity(), hit.matchedQuestion());
                        }
                    }
                }

                if (!any) {
                    writer.println("  (무관 판정 실패 없음)");
                }
            }
        }
        return outputFile;
    }

    private List<CSVRecord> readCsv(String resourceName) throws IOException {
        try (var inputStream = getClass().getClassLoader().getResourceAsStream(resourceName)) {
            if (inputStream == null) {
                throw new IllegalStateException("리소스를 찾을 수 없습니다: " + resourceName);
            }
            var reader = new InputStreamReader(inputStream, StandardCharsets.UTF_8);
            try (var parser = CSVFormat.DEFAULT.builder()
                    .setHeader().setSkipHeaderRecord(true).build().parse(reader)) {
                return parser.getRecords();
            }
        }
    }
}