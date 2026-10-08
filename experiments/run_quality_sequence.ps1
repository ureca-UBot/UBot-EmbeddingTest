$ErrorActionPreference = 'Stop'
$taskStages = @(
    @('B'),
    @('C'),
    @('legacy', '--legacy-key', 'legacy_60'),
    @('legacy', '--legacy-key', 'legacy_235'),
    @('D', '--reranker', 'bge', '--view', 'question'),
    @('D', '--reranker', 'bge', '--view', 'question_answer'),
    @('D', '--reranker', 'jina', '--view', 'question'),
    @('D', '--reranker', 'jina', '--view', 'question_answer')
)
foreach ($taskStage in $taskStages) {
    $taskService = if ($taskStage -contains 'jina') { 'jina-benchmark' } else { 'benchmark' }
    & docker compose --profile quality run --rm $taskService model_benchmarks/scripts/run_retrieval_benchmark.py @taskStage
    if ($LASTEXITCODE -ne 0) {
        throw "Retrieval stage failed: $($taskStage -join ' ')"
    }
}
