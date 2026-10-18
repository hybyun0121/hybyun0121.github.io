---
layout: "writing"
writing_type: "ai_notes"
title: "인과 그래프로 \"조립 가능한\" 의사결정 환경 만들기: SCDM과 SCDP"
subtitle: "The Design and Composition of Structural Causal Decision Processes"
date: "2026-10-18"
permalink: "/posts/structural-causal-decision-processes/"
excerpt: "문제: 부분 시스템의 조립, 의사결정 제약, 정보 처리 자원의 한계를 함께 표현할 모델이 필요하다. 방법: root와 decision constraint를 갖는 SCDM 및 이를 기간별로 반복하는 SCDP를 정의한다."
source_read_date: "2026-10-18"
source_note_id: "2026-05_structural-causal-decision-processes"
source_paper: {"title": "The Design and Composition of Structural Causal Decision Processes", "authors": "Sebastian Benthall, Alan Lujan", "venue": "arXiv preprint (2026-05-04), cs.CE / cs.AI / cs.GT / econ.TH", "url": "https://arxiv.org/abs/2605.02681"}
tags: ["agent", "causal-model", "decision-theory", "computational-economics", "resource-rationality", "formalism"]
takeaways: ["문제: 부분 시스템의 조립, 의사결정 제약, 정보 처리 자원의 한계를 함께 표현할 모델이 필요하다.", "방법: root와 decision constraint를 갖는 SCDM 및 이를 기간별로 반복하는 SCDP를 정의한다.", "결과: 분해·최적화와 표현력에 관한 형식적 결과이며, 새로운 실험 점수나 경험적 검증을 보고한 논문은 아니다."]
ai_tool: "Claude"
editor_note: "AI 리딩 노트를 블로그 형식으로 재구성했습니다. 논문의 주장·평가 조건과 AI 해석을 구분해 읽어 주세요."
---

## 문제와 배경

### 배경과 읽는 관점

복잡한 의사결정 시스템을 한 번에 모델링하면, 작은 부품을 고치거나 시스템 일부의 행동을 검증하기도 어려워진다. 특히 사람과 AI의 정보 처리 자체가 제한된 자원을 소비한다면, 모든 정보를 공짜로 관측하는 의사결정자만으로는 관심 문제를 표현하기 부족할 수 있다. 이 논문은 부분 모델을 조립하면서 그런 제약을 모델 안에 넣는 형식 체계를 제안한다.

### 기존 접근과 연구 질문

- 오늘날 컴퓨팅 시스템은 기술 컴포넌트 + 사람 + AI agent가 섞인 부분 시스템들의 조합이고, 이들 사이에는 데이터 흐름이 오간다. 현실적인 모델이라면 **분해 가능(decomposable)**해야 하고 **인지 자원의 한계가 모델 내부(endogenous)에** 들어가야 한다고 저자들은 말한다(서론).
- 기대하는 응용: **policy simulation**(디지털 경제 정책 효과), **system design / mechanism design**(데이터 흐름의 파라미터 설계), **digital twin**(전략적 행위자 + PINN 결합).
- 끌어오는 연구 흐름 네 가지(§2):
  1. **경제학 / ABM**: 마이크로데이터와 연산이 싸지면서 "rational expectations" 같은 강한 가정을 풀고 직접 시뮬레이션하는 쪽으로 가는 흐름(Axtell & Farmer 2025). 그래도 decision-theoretic하게 잘 정의된 모델이 필요하다는 입장.
  2. **CS / AI safety의 causal influence diagram**: MAID(Koller & Milch 2003) → SCIM / Structural Causal Game(Hammond et al.). reward tampering, deception, manipulation 등을 이 틀로 분석해 왔다.
  3. **Composability**: 최적화(MAID의 subgame 분해, EGM 다단계 분해, factored/hierarchical MDP)와 시뮬레이션(부품 단위 검증) 양쪽에서 필요.
  4. **Resource rationality**(Lieder & Griffiths 2020)와 **variable discounting**(Strotz, Laibson의 β-δ, Stachurski의 state-dependent discount, Krusell-Smith의 이질적 β).
- 저자들이 제시하는 위치(Figure 1. 그림이 모두 TikZ로 그려져 있어 별도 figure 파일이 없음 → [paper.pdf](https://arxiv.org/pdf/2605.02681)에서 확인): Hammond et al.의 taxonomy(BN→CBN→SCM / ID→CID→SCIM / MAID→CG→SCG)의 세로축(associational → interventional → counterfactual)에 **functional(SCDM)**, **dynamic(SCDP)** 두 층을 추가. 이 논문은 **single-agent** 열만 다룬다.

이 글의 대상은 새로운 agent benchmark의 점수가 아니라 모델의 표현력과 분해 조건이다. 무엇을 접합부로 삼아 두 모델을 연결하는지, 어떤 그래프 조건에서 부분 문제의 해를 전체 최적화에 사용할 수 있는지가 중심 질문이다.

## 핵심 아이디어

### 먼저 직관으로 이해하기

분포와 구조방정식을 아직 정하지 않은 root 변수를 부분 모델의 입력 소켓으로 사용한다. 다른 모델의 출력이 이 root를 채우도록 연결하면 composition을 표현할 수 있다. 이후 적절한 독립성 조건 아래에서 뒤쪽 부분 문제의 value를 먼저 계산하고, 그 값을 앞쪽 의사결정에 전달하는 구조를 만든다.

### SCDM: SCIM과 무엇이 다른가 (§3.2, Definition "SCDM")

SCDM = `(V, Z, ℰ, (X, D, U), Pr, f, Γ, θ)`
- `V`: endogenous 변수 — state `X`, decision `D`, utility `U`로 분할
- `Z`: exogenous noise (부모 없음, 분포 `Pr(θ)`가 주어짐)
- `f`: decision이 아니고 부모가 있는 변수에 대한 **결정적 structural function**
- `Γ_D`: decision constraint — 부모 값에 따라 허용 행동 집합을 정함 (예: 예산 제약 `c₁ ≤ b₁`)

SCIM과의 차이(저자 정리):

| 항목 | SCIM | SCDM |
|---|---|---|
| 행동 제약 | 없음 | `Γ` 명시 (control theory의 binding constraint와 결합) |
| 변수 관계 | deterministic CPD | 결정적 함수 `f` (스타일 차이) |
| 부모 없는 state 변수 | 불가 | **root variable 허용** — 분포도 구조방정식도 없음 |
| noise | 변수마다 `E_V` | exogenous noise를 별도 노드로 표기 |
| 지향 | 이산/연속 일반 | 연속 state/shock/decision, 미분·역함수 가능 `f`를 염두 (EGM 류 해법) |

핵심은 **root**: SCDM은 "root 값 → 나머지 모든 변수"로 가는 확률적 함수로 볼 수 있고, root가 두 모델을 붙이는 "소켓" 역할을 한다.

### Decision rule과 root value function (§3.2.2)
- decision rule `π_D : dom(Pa_D) → dom(D)`, 단 `π_D(p) ∈ Γ_D(p)`. 모든 D에 대한 rule 묶음이 policy profile `π`.
- policy profile + root 값이 주어지면 SCDM은 완전한 SCM이 된다.
- root value function (Eq. 1): `v_M(x̃) = max_π E_Z[ Σ_U f*_U(x̃, z, π) ]`
- 저자들은 **decision rule에 대한 최적화**와 **행동(action)에 대한 최적화**(Eq. 2, 전지적 agent의 `d*`)를 명시적으로 구분한다. rule은 `Pa_D`라는 제한된 정보만 보므로, 최적 행동을 보장하는 rule이 존재하지 않을 수 있다.

### Composition (§3.2.3, Definition "SCDM Composition")
- 두 SCDM `M₁`, `M₂`에서 공유 변수 `Y = V₁ ∩ V₂ ⊆ X̃₂`(M₂의 root)이면, 집합들을 합집합으로, 분포를 곱으로 묶어 `M₁ ∘ M₂`를 정의한다. `Y`가 **bridge**.
- 두 SCDM의 composition은 SCDM이다. 역으로 하나를 둘로 쪼개는 것이 decomposition. 서로 완전히 분리된 두 모델의 composition도 허용되는 일반적 정의다.

### Sequential decomposition과 orthomodularity (§3.3)
- **Orthomodularity (Definition "Orthomodularity")**: bridge `Y`와 `U₂`가 `D₂ ∪ Pa_{D₂}`가 주어졌을 때 d-separated. (충분조건이지 필요조건은 아니라고 명시.)
- **Theorem (non-reliant decomposition)**: orthomodular decomposition이면 어떤 `D₂`도 `D₁`에 **strategically rely**하지 않는다. 증명은 Koller & Milch의 s-reachability → soundness 정리에 기댄다(부록 A.2).
- 그 결과 `M₂`의 **bridge value function** `v_{M₂}(y)`(Eq. 3)를 continuation value로 쓸 수 있고, 전체 문제가 Eq. 4처럼 "1단계 utility + `v_{M₂}`(bridge 값)"의 최대화로 바뀐다 — 즉 backward induction.
- 그림: Figure 3 ([paper.pdf](https://arxiv.org/pdf/2605.02681)) — `Y → U₂`로 가는, `Pa_{D₂} ∪ D₂`에 막히지 않는 경로(빨간 점선)가 있으면 orthomodularity가 깨진다.

### SCDP (§4)
- 출발점: Maliar et al.의 **DSOP**(Definition "DSOP": 제약된 행동, 결정적 transition `g`, 할인된 기대 보상 최대화).
- 저자들은 shifted Bellman equation( 기대값이 max 바깥, shock이 행동 이전에 실현) 형태를 쓰고, 한 기간의 해를 구성하는 요소 묶음을 **P-block** `(𝒳, 𝒵, P_𝒵, 𝒜, A, g, r, β_t)`로 정의. 여기서 `β`는 상수가 아니라 `𝒳×𝒵×𝒜 → ℝ` 함수로 허용된다.
- **Theorem (SCDM ↔ P-block)**: SCDM + discount 변수 `β ∈ V` + root → end-of-period 변수 매핑 `w`가 있으면 대응되는 P-block을 만들 수 있다 (구성적 증명: `𝒳 ⊆ X̃`, `𝒜 = D`, `A = Γ`, `g = f_W`, `r = Σ f*_U`, `β = f*_β`).
- **SCDP (Definition "SCDP")** = `(M, β, w)`를 P-block으로 묶어 만든 DSOP. 즉 "**SCDM을 매 기간 반복하고 다음 기간 root에 이번 기간 끝 변수를 꽂은 것**".

### 정의와 수식

- Root value: `v_M(x̃) = max_π E_Z[Σ_{U∈U} f*_U(X̃ = x̃, z, π)]` (Eq. 1)
- Bridge value: `v_{M₂}(y) = max_{π₂} E_Z[Σ_{U∈U₂} f*_U(y, z, π₂)]` (Eq. 3)
- 분해된 해: `v_{M₀}(x̃) = max_{π₁} E_{Z₁}[Σ_{U₁} f*_{U₁}(x̃, z₁, π₁) + v_{M₂}(f_Y(x̃, z₁, π₁))]` (Eq. 4)
- Shifted Bellman (§4.2): `v_t(x_t) = E_{z_t}[ max_{a_t∈A_t(x_t,z_t)} { r_t(x_t,z_t,a_t) + β_t v_{t+1}(x_{t+1}) } ]`
- 복잡도 (§3.4, §4.2.2):
  - 정적: naive `O(|π₀|)`, `|π₀| = |π₁||π₂|` → 분해 시 `O(|π₂||Y| + |π₁|)`
  - 동적 1회 iteration: naive `O(|𝒳||𝒵||𝒜|)` → 분해 시 `O(|X̃||𝒵|(|π₂||Y| + |π₁|))`

## 이론적 결과와 적용 범위

### 적용 범위와 설정

**실험 없음.** 모든 논증은 정의·정리·복잡도 분석과 경제학 toy 예제로 이뤄진다. 구현체로 오픈소스 라이브러리 **`scikit-agent`**(https://github.com/scikit-agent/scikit-agent)를 언급하지만, 논문 안에서 이를 써서 측정한 결과는 없다.

예제 목록:

| § | 예제 | 보여주려는 것 |
|---|---|---|
| 3.1 | Fisher 2기간 소비 | `a₁`이 bridge. `c₂`는 사실상 결정됨(`c₂ = b₂ + y₂`) → 1차원 continuation value `v₂(b₂)` |
| 3.5 | 2기간 소비 + **habit formation** (`h₂ = c₁`) | `c₁`이 `a₁`과 `h₂` 두 경로로 2기간에 영향 → bridge를 `{b₂, h₂}`로 **확장**해야 orthomodular. continuation value가 2차원 `v₂(b₂, h₂)` (Eq. 5) |
| 4.1 | 소비-저축 + **portfolio allocation** `α` | 두 모듈을 compose. bridge = `{a, r}`. allocation 모듈엔 utility 노드가 없어 "trivially orthomodular" |
| 5.1 | **latent income process** (+ 선택적 2-state memory `m'`) | 정보 집합이 `Pa_D`로 제한됨, belief filter가 필수가 아님 |
| 5.5 | **costly memory** (decision 3개: belief update `q`, 소비 `d`, 기억 정밀도 `r`, 비용 `v = −1/r`) | memory와 belief 형성을 decision + 비용으로 내재화 → "POMDP로는 모델링 불가" |
| 6.1 | **stochastic discount factor** (Krusell-Smith 류, `β ∈ {β_low, β_high}` Markov chain) | `β`를 state 변수로 두어도 orthomodularity 유지 |

> 실험 결과가 없으므로 "Finding"은 저자들의 이론적 주장 단위로 정리한다.

### Finding 1 — Root variable이 composition을 가능하게 한다
분포·구조방정식이 없는 root를 허용하면, 한 모델의 출력 변수를 다른 모델의 root에 꽂는 방식으로 SCDM끼리 붙일 수 있고, 결과도 SCDM이다(§3.2.3). 저자들은 결론에서 이 root가 SCDM을 SCIM과 구분 짓는 핵심이며, composition과 SCDP 변환을 모두 가능케 한다고 말한다.

### Finding 2 — Orthomodularity ⇒ 분리 가능한 subgame ⇒ backward induction
bridge와 후속 utility가 후속 decision과 그 부모로 d-separate되면 후속 decision이 앞선 decision에 strategically rely하지 않는다(§3.3 Theorem). 그래서 뒤 컴포넌트를 bridge의 각 값에 대해 먼저 풀고, 앞 컴포넌트는 그 value function만 참조하면 된다(Eq. 4). 탐색 비용은 곱(`|π₁||π₂|`)에서 합(`|π₂||Y| + |π₁|`)으로 줄어든다(§3.4). 동적 설정의 value iteration 1회에도 같은 이득이 그대로 적용된다고 주장한다(§4.2.2).

### Finding 3 — 분해는 실패하지 않고 bridge가 커질 뿐이다
habit formation처럼 과거 결정이 여러 경로로 미래에 영향을 주면 bridge를 그 경로들을 모두 담는 sufficient statistic으로 확장해야 한다. 이 경우 분해 자체는 유지되지만 bridge(=continuation value 정의역)의 차원이 늘어난다(§3.5 "General lessons").

### Finding 4 — SCDM 한 기간 = P-block 한 개, 반복하면 SCDP
discount 변수와 end-state 매핑만 추가하면 SCDM에서 P-block을 기계적으로 만들 수 있고(§4.2.1 Theorem), 이를 반복하면 DSOP가 된다. 그러니까 "dynamic programming의 한 기간"을 인과 그래프로 그린 것이 SCDM이다.

### Finding 5 — SCDP ⊋ MDP, POMDP (저자 주장, 비형식적)
- **vs MDP**: MDP의 policy는 `π(s)`로 전체 state를 본다. SCDP의 rule은 `Pa_D`만 보므로 partial observability와 latent 변수, 그리고 결정 사이에 shock이 끼는 구조를 표현할 수 있다(§5.2).
- **vs POMDP**: POMDP는 보통 Bayesian filter로 sufficient statistic을 담은 belief-state MDP를 암묵적으로 가정한다. SCDP는 과거→현재 정보 흐름을 **요구하지 않고**, 필요하면 memory를 변수·decision으로 **명시적으로 넣는다**(§5.3). 한 기간 안에 정보 집합이 서로 다른 decision을 여러 개 둘 수도 있다.
- **costly memory 예제**(§5.5, Figure 7): 기억 정밀도 `r`을 고르고 그 비용 `v = −1/r`을 utility로 받는다. 저자들은 이것이 "well-formulated SCDP이며 POMDP로는 모델링할 수 없다"고 단언한다. 이 예제가 resource rationality를 표현할 수 있다는 근거다.

### Finding 6 — Variable discounting은 그냥 state 변수 하나
`β ∈ V`이므로 자체 구조방정식(quasi-hyperbolic, state-dependent `β(x)`, Markov `β`)을 가질 수 있다. Krusell-Smith 류 예제(§6.1 Bellman equation, Figure 8)에서 `β'`는 `β`와 shock에만 의존하고 endogenous choice에 의존하지 않으므로 orthomodularity가 유지되고 recursive 구조가 보존된다(§6.1). 반대로 그래프 구조가 variable discounting이 orthomodularity를 *깨는* 경우도 드러내 줄 수 있다고 언급한다.

## 한계와 해석

### 저자들이 밝힌 한계

- **Single-agent만** 다룬다. multi-agent(SCG로의 확장)는 future work(서론, 결론).
- **해법 알고리즘과 데이터 fitting 알고리즘은 아직 없다**. "We do not guarantee that all environments developed in this way are efficiently soluble"(§5.4). 관심사는 효율적 풀이보다 *유연한 기술(description)*이다.
- Orthomodularity는 **충분조건일 뿐 필요조건이 아니다**(Orthomodularity 정의 직후).
- 측도론적 기초는 다루지 않는다(§3.2). utility는 실수값이라고 가정.
- 소스에 남은 저자 TODO 주석: "t와 t+1 모델 사이의 관계가 orthomodular여야 하는가?"(SCDP 정의 직후 주석) → 동적 경우의 분해 조건이 아직 정리되지 않았음을 시사.

### AI의 해석

<aside class="note-callout" markdown="1">
**아래는 AI 정리자의 해석이며 원논문의 주장과 구분합니다.**
</aside>

> 이 섹션은 논문의 주장이 아니라 정리자(Claude)의 해석이다.

1. **"POMDP보다 엄밀히 표현력이 크다"는 주장의 근거가 약하다.** 초록에 "strictly more expressive"라고 했지만 본문은 예제와 서술 수준이다. memory 변수를 environment state에 넣고 memory 갱신을 action으로 두면, costly memory 예제도 *augmented state POMDP*나 **finite-state controller**(제한된 memory를 가진 POMDP policy)로 상당 부분 재구성할 수 있어 보인다. 실제 차이는 "agent policy가 `Pa_D`로 제한된다"는 점, 즉 *policy class 제약*이고, 이건 influence diagram 쪽의 **LIMID**(Lauritzen & Nilsson 2001, limited-memory influence diagram)가 이미 다룬 개념이다. 그런데 LIMID 인용이 없다. "모델 클래스의 표현력"과 "허용 policy class의 차이"를 구분해서 형식적으로 증명했어야 한다.
2. **Orthomodularity 정의가 다소 미묘하다.** 많은 예제에서 bridge `Y`가 곧 `Pa_{D₂}`의 원소다(예: `b₂ → c₂`). 그러면 `Y` 자체를 조건부로 두고 d-separation을 묻는 셈이라, 조건이 쉽게 성립하거나 정의가 모호해진다. §3.3 Theorem의 증명도 "`D₁`에서 `U₂`로 가는 모든 경로는 bridge를 지난다"는 사실에 크게 의존하지만, `D₁`이 `Pa_{D₂}`에 영향을 주어 `D₂`의 *정보*를 바꾸는 경로는 따로 다루지 않는다.
3. **Portfolio 예제는 정의와 잘 맞지 않는다.** bridge `{a, r}`에서 `a`는 소비→allocation, `r`은 allocation→소비 방향으로 흐른다. 서로가 서로의 root를 받는 구조라서 Composition 정의의 "`Y ⊆ X̃₂`"(한쪽 모델의 root에만 붙는다)를 그대로 만족하지 않는다. 본문도 "α를 먼저 풀고 그다음 c"라고 해서 composition 순서와 해법 순서가 반대다. 사실상 한 기간 안의 3단계(소비 → allocation → 다음 기간) 구조로 보는 게 맞아 보인다.
4. **복잡도 분석이 거칠다.** `|π|`는 *decision rule 전체 집합*의 크기인데, bridge의 각 값 `y`마다 `M₂`를 푸는 단계에서는 그 `y`에 대해 *행동*만 고르면 된다. 그래서 `|π₂||Y|`라는 표현은 단위가 섞여 있다. 또 bridge value 계산을 `O(1)`로 가정하고, continuation value의 근사·보간 비용(연속 상태에서 가장 큰 비용)은 다루지 않는다. Eq. 4 마지막 줄의 `v_{M₂}(..., π₁)`처럼 정의(Eq. 3)와 맞지 않는 표기도 있다.
5. **문서 완성도.** SCDM 정의(Definition 1)의 본문 이름이 "structural equation influence model"로 되어 있고, SCDP 정의는 SCDP를 "a SCIM M"으로 정의하는 등 용어 혼용이 있다. Γ도 튜플 표기에서 빠져 있다. 소스에 TODO와 주석 처리된 문단이 남아 있어 워크숍/초기 버전 성격이 강하다(파일명 `esif2026.tex`).
6. **그럼에도 가치 있는 지점.** (a) 경제학의 Bellman/EGM 전통과 CS의 causal influence diagram 전통을 *같은 그림*으로 이은 것, (b) "bridge = sufficient statistic, 분해 가능성 = 그래프 조건"이라는 관점, (c) memory·attention·discount를 **decision 변수 또는 state 변수로 그래프에 올려** 비용과 함께 최적화 대상으로 만드는 모델링 습관. 이 셋은 LLM agent 환경 설계에도 그대로 옮겨 쓸 수 있는 사고 틀이다.

## 더 생각해볼 질문

아래는 AI가 정리 과정에서 제안한 후속 질문입니다. 본인이 직접 작성한 의견이나 논문에서 검증한 결과로 표시하지 않습니다.

- **LLM agent의 memory를 decision 변수로**: context에 무엇을 남길지(summarize/retrieve/forget), 비용(token, latency)은 얼마인지를 costly memory 예제(§5.5)처럼 `q`, `r`, `v` 노드로 그리면 "memory 정책"을 학습 대상으로 명시할 수 있지 않을까? 이것은 Agent Plasticity류 self-improvement 평가와 어떻게 연결될까?
- **Tool-using agent 환경의 composability**: 각 tool/subsystem을 SCDM 컴포넌트로, 그 사이의 데이터 흐름을 bridge로 보면, orthomodularity가 "어떤 sub-task는 독립적으로 RL 학습해도 된다"는 조건 역할을 할 수 있을까? 이는 Generalization Tax 논문의 cross-domain 학습 질문과도 맞닿는다.
- **Environmental novelty**: 환경 변화가 그래프의 어느 부분(structural function `f`? bridge? `Pa_D`?)에 생기느냐에 따라 적응 난이도가 달라진다고 볼 수 있을까?
- 그래프 조건(d-separation)으로 **credit assignment를 분해**하는 것이 실제 deep RL에서 sample efficiency 이득으로 이어지는지 실험해볼 수 있을까?

## 참고자료

- [원논문](https://arxiv.org/abs/2605.02681)
- [논문 PDF](https://arxiv.org/pdf/2605.02681)

<details markdown="1">
<summary>용어 정리</summary>

| 용어 | 뜻 |
|---|---|
| **SCM** | Structural Causal Model. 모든 확률성이 exogenous root에 있고 나머지는 구조방정식으로 결정되는 Pearl식 인과 모델 |
| **SCIM / SCG** | Structural Causal Influence Model(1인) / Structural Causal Game(다인). SCM에 decision·utility 변수를 더한 것 (Hammond et al.) |
| **SCDM** | 이 논문의 Structural Causal Decision Model. SCIM + decision constraint `Γ` + root variable |
| **SCDP** | Structural Causal Decision Process. SCDM + discount 변수 `β` + end-state 매핑 `w`를 반복한 동적 문제 |
| **root variable** | 부모도, 분포도, 구조방정식도 없는 state 변수. 모델의 "입력 소켓" |
| **bridge** | composition에서 두 모델이 공유하는 변수 집합 `Y` |
| **orthomodularity** | `Y`와 `U₂`가 `D₂ ∪ Pa_{D₂}` 조건부로 d-separated. 순차 분해의 충분조건 |
| **strategic reliance** | `D'`의 rule을 바꾸면 `D`의 최적 rule이 바뀌는 관계 (Koller & Milch) |
| **s-reachability** | strategic reliance의 그래프 기반 판정 기준 |
| **decision rule vs action** | rule은 `Pa_D`만 보는 함수. action 최적화는 전지적 agent 기준 |
| **DSOP** | Dynamic Stochastic Optimization Problem (Maliar et al.) |
| **P-block** | 한 기간의 Bellman 업데이트에 필요한 요소 묶음 `(𝒳, 𝒵, P_𝒵, 𝒜, A, g, r, β)` |
| **shifted Bellman equation** | shock 기대값을 max 바깥에 둔 Bellman 형태 (shock 관측 후 행동) |
| **resource rationality** | 인지 자원의 비용을 utility에 명시적으로 넣고 그 하에서 최적화한다고 보는 인지 모델링 관점 |
| **habit formation** | 과거 소비가 현재 효용의 기준점이 되는 선호 (`∂u/∂h < 0`) |
| **EGM** | Endogenous Grid Method. 소비-저축 문제의 효율적 해법 (Lujan의 EGMⁿ이 다단계 분해 버전) |

</details>
