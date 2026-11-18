`sample_submission` 폴더는 **SWE(Software Engineering) 이슈 해결을 위한 멀티 에이전트(메인 에이전트 + 코드 분석 서브 에이전트) 베이스라인 제출 템플릿**입니다. 총 6개의 파일로 구성되어 있으며, YAML의 `!include` 지시자를 통해 모듈화되어 있습니다.

---

### 1. 전체 구조 및 연결 관계

```text
sample_submission/
├── agent.yaml                  # [진입점] 메인 에이전트 정의 (프롬프트, 툴, 서브 에이전트, 샘플링 연결)
├── eval_config.yaml            # 평가 환경(Stage 1) 실행 예산 및 타임아웃 설정
├── configs/
│   └── sampling.yaml           # LLM 생성 하이퍼파라미터 및 추론(Thinking) 설정
├── prompts/
│   ├── system.md               # 메인 에이전트 시스템 프롬프트
│   └── analyzer.md             # 서브 에이전트 시스템 프롬프트
└── sub_agents/
    └── code_analyzer.yaml      # 코드 분석 전용 서브 에이전트 정의
```

---

### 2. 파일별 상세 역할 분석

#### ① [agent.yaml](file:///c:/Users/KOSCOM/Documents/Workspace/Google-TheGemma4DeveloperAgentCompetition/sample_submission/agent.yaml) — 메인 에이전트 설정 (Root Config)
전체 에이전트 시스템의 진입점 역할을 하는 핵심 설정 파일입니다.
* **모델(`model`)**: `gemma-4-12b-it-qat-w4a16-ct` (양자화된 Gemma 4 12B Instruct 모델)을 지정합니다.
* **시스템 프롬프트(`instruction`)**: `!include`를 통해 [prompts/system.md](file:///c:/Users/KOSCOM/Documents/Workspace/Google-TheGemma4DeveloperAgentCompetition/sample_submission/prompts/system.md)를 불러옵니다.
* **사용 도구(`tools`)**:
  * **실행 및 파일 조작**: `run_command`, `read_file`, `edit_file`, `write_file`, `get_status`, `submit_patch`
  * **코드 그래프/시맨틱 검색**: `get_code_neighbors`, `search_similar_code`, `get_code_subgraph`
  * **서브 에이전트 호출(`agent_tool`)**: [sub_agents/code_analyzer.yaml](file:///c:/Users/KOSCOM/Documents/Workspace/Google-TheGemma4DeveloperAgentCompetition/sample_submission/sub_agents/code_analyzer.yaml)을 도구처럼 호출할 수 있게 연결하며, `skip_summarization: true`를 통해 서브 에이전트의 응답을 요약 없이 그대로 전달받습니다.
* **생성 설정(`generate_content_config`)**: [configs/sampling.yaml](file:///c:/Users/KOSCOM/Documents/Workspace/Google-TheGemma4DeveloperAgentCompetition/sample_submission/configs/sampling.yaml)을 불러옵니다.

#### ② [eval_config.yaml](file:///c:/Users/KOSCOM/Documents/Workspace/Google-TheGemma4DeveloperAgentCompetition/sample_submission/eval_config.yaml) — 평가 실행 제약 조건 설정
Stage 1 추론 평가 시 각 태스크당 부여되는 리소스 예산(Budget)과 타임아웃을 제어합니다.
* `timeout_seconds: 60` — 개별 샌드박스 명령어 실행 제한 시간 (60초)
* `max_tool_calls: 10` — 태스크당 최대 도구 호출 횟수 (10회)
* `max_time_minutes: 1` — 태스크당 전체 제한 시간 (1분)
* `max_turns: 50` — 최대 대화 턴 수 (50턴)
> **참고**: 베이스라인에 설정된 예산(`max_tool_calls: 10`, `max_time_minutes: 1`)이 매우 타이트하므로, 대회 허용 범위 내에서 이 값들을 조정할 수 있는지 확인하거나 프롬프트가 이 제한에 맞춰 빠르게 패치를 제출하도록 설계되어 있습니다.

#### ③ [configs/sampling.yaml](file:///c:/Users/KOSCOM/Documents/Workspace/Google-TheGemma4DeveloperAgentCompetition/sample_submission/configs/sampling.yaml) — LLM 디코딩 및 추론(Thinking) 파라미터
메인 에이전트와 서브 에이전트가 공통으로 사용하는 생성 파라미터입니다.
* `temperature: 0.2`, `top_p: 0.95` — 코드 수정의 일관성과 정확도를 높이기 위해 낮은 온도를 사용합니다.
* `max_output_tokens: 16384` — 최대 출력 토큰 수입니다.
* `thinking_config` — `thinking_budget: 4096` 및 `include_thoughts: true`로 설정되어 있어, 응답 및 도구 호출 전 최대 4,096 토큰 내에서 내부 추론(Chain-of-Thought)을 수행합니다.

#### ④ [prompts/system.md](file:///c:/Users/KOSCOM/Documents/Workspace/Google-TheGemma4DeveloperAgentCompetition/sample_submission/prompts/system.md) — 메인 에이전트 행동 지침
제한된 시간/턴(`eval_config.yaml`의 제약) 내에 문제를 해결하도록 강제하는 메인 프롬프트입니다.
* **핵심 목표**: 8~10턴 이내의 최소한의 도구 호출로 문제를 파악하고 수정 후 제출.
* **4단계 워크플로우**:
  1. **타겟 파일 즉시 식별**: 전체 레포지토리에 `find`/`grep`을 남발하지 않고 문제 설명의 키워드나 `search_similar_code`를 활용.
  2. **직접적인 코드 수정**: `edit_file` 등으로 최소한의 변경만 적용.
  3. **타겟 테스트만 실행**: 전체 `pytest` 실행을 엄격히 금지(타임아웃 방지)하고, 수정한 기능과 직결된 테스트 파일/메서드만 지정해 실행. 기존에 깨져 있던 무관한 테스트는 무시.
  4. **즉시 패치 제출**: 타겟 테스트 통과 즉시 `submit_patch` 호출(`patch_size > 0` 확인).
* **금지 사항(Anti-Patterns)**: 테스트 파일(`tests/*`, `test_*.py`) 수정 절대 금지(자동 실격 처리), `/workspace` 외부 탐색 금지, 빈 패치 종료 금지.

#### ⑤ [sub_agents/code_analyzer.yaml](file:///c:/Users/KOSCOM/Documents/Workspace/Google-TheGemma4DeveloperAgentCompetition/sample_submission/sub_agents/code_analyzer.yaml) — 코드 분석 서브 에이전트 설정
메인 에이전트의 컨텍스트와 턴을 아끼고, 코드 탐색만 전담시키기 위해 분리된 **읽기 전용(Read-Only) 서브 에이전트**입니다.
* **도구 제한**: 파일 수정이나 명령어 실행 권한 없이 `read_file`, `search_similar_code`, `get_code_neighbors`, `get_code_subgraph` 등 **조회 및 코드 그래프 탐색 도구 4가지만** 보유합니다.
* 모델과 샘플링 설정은 메인 에이전트와 동일하며, 프롬프트로 [prompts/analyzer.md](file:///c:/Users/KOSCOM/Documents/Workspace/Google-TheGemma4DeveloperAgentCompetition/sample_submission/prompts/analyzer.md)를 사용합니다.

#### ⑥ [prompts/analyzer.md](file:///c:/Users/KOSCOM/Documents/Workspace/Google-TheGemma4DeveloperAgentCompetition/sample_submission/prompts/analyzer.md) — 서브 에이전트 행동 지침
`code_analyzer_agent`에게 부여된 간결한 역할 지침입니다.
* 코드 검색 및 심볼 그래프(`get_code_neighbors`, `get_code_subgraph`) 도구를 사용해 버그의 원인이 되는 정확한 함수/클래스/라인 범위를 찾습니다.
* 메인 에이전트에게 **① 수정할 정확한 파일 경로와 라인 번호, ② 근본 원인(Root cause) 설명, ③ 권장하는 최소한의 코드 변경안**을 구조화하여 반환하도록 지시합니다.