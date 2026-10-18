---
layout: "writing"
writing_type: "ai_notes"
title: "Agent RL의 \"일반화 세금\": 어떤 환경에서 RL을 돌려야 덜 내는가"
subtitle: "Paying Less Generalization Tax: A Cross-Domain Generalization Study of RL Training for LLM Agents"
date: "2026-10-18"
permalink: "/posts/generalization-tax-agent-rl/"
excerpt: "문제: 한 환경에서 agent RL을 수행하면 다른 도메인 성능을 잃는 generalization tax가 발생한다. 방법: 네 환경의 single-source 전이와 state 정보량·계획 복잡성, SFT·thinking 조건을 비교한다."
source_read_date: "2026-10-17"
source_note_id: "2026-01_generalization-tax-agent-rl"
source_paper: {"title": "Paying Less Generalization Tax: A Cross-Domain Generalization Study of RL Training for LLM Agents", "authors": "Zhihan Liu*, Lin Guan*, Yixin Nie, Kai Zhang, Zhuoqun Hao, Lin Chen, Asli Celikyilmaz, Zhaoran Wang†, Na Zhang† (*공동 1저자, †공동 교신)", "venue": "arXiv preprint (2026-01-26)", "url": "https://arxiv.org/abs/2601.18217"}
tags: ["agent", "rl", "generalization", "post-training"]
takeaways: ["문제: 한 환경에서 agent RL을 수행하면 다른 도메인 성능을 잃는 generalization tax가 발생한다.", "방법: 네 환경의 single-source 전이와 state 정보량·계획 복잡성, SFT·thinking 조건을 비교한다.", "결과: 환경 선택과 정보 증강이 OOD 보존에 영향을 주지만, 네 환경의 비교를 보편적 인과 법칙으로 해석할 수는 없다."]
editor_note: "AI 리딩 노트를 블로그 형식으로 재구성했습니다. 논문의 주장·평가 조건과 AI 해석을 구분해 읽어 주세요."
---

## 문제와 배경

### 배경과 읽는 관점

Agent의 학습 환경을 아무리 잘 만들더라도 실제 배포에서 만날 도구·상태·목표를 모두 포함하기는 어렵다. 한 환경에서의 reward를 높이는 동안 다른 환경에서 유용했던 능력이 유지되는지는 별도의 문제다. 이 논문은 더 많은 학습 자체보다, 제한된 환경 중 무엇을 선택해 학습할 것인가에 초점을 맞춘다.

### 기존 접근과 연구 질문

- Generalist LLM agent의 post-training은 **좁은 범위의 task**로 이루어지지만 배포 범위는 훨씬 넓다. 데이터 flywheel을 잘 돌려도 sampling → 후처리 → 대규모 학습 사이클이 비싸서, 한 번에 넣을 수 있는 도메인은 제한적이다(§1).
- Online RL에는 reward가 있고 multi-rollout이 가능한 **interactive 환경**이 필요하다. 이런 환경은 만들기 비싸고, proprietary 도메인에서는 만들 수조차 없다. 공개 agent 벤치마크도 대부분 특정 사이트나 워크플로에 한정된다(§1).
- 그래서 저자들은 **data-centric 질문**을 던진다. "RL 학습용 task를 어떻게 고르거나 어떤 simulator를 선택적으로 만들어야, 보지 못한 도메인으로 agent 능력이 최대한 보존·전이되는가?"
- 일반화에는 두 수준이 있다고 구분한다(§1, §3.2).
  - **In-domain(ID)**: 같은 환경 안의 held-out task. 예: Amazon 쇼핑 → eBay 쇼핑.
  - **Cross-domain(OOD)**: 도구·state·goal이 크게 다른 환경. 예: 쇼핑 → 수학 문제 풀이.
  - 기존 "RL은 일반화된다"는 증거는 대부분 ID 수준이다. 이 논문은 OOD를 본다. 예비 실험에서는 RL이 진행될수록 ID는 오르고 held-out 환경 점수는 떨어졌다. 그래서 이 연구는 **예방(prevention) 지향**이다. OOD 성능을 *올리기*보다 *덜 잃는* 선택지를 찾는 것이 목표다.

읽을 때는 같은 환경의 새 문제에 잘 대응하는 ID 일반화와, 도구와 목표까지 달라지는 cross-domain 전이를 나눈다. 여기서 말하는 세금은 학습한 환경에서 얻은 이득의 대가로 다른 도메인의 성능을 잃는 현상이며, 연구 목표는 그 손실을 덜 내는 조건을 찾는 것이다.

## 핵심 아이디어

### 먼저 직관으로 이해하기

여러 환경을 한꺼번에 섞기보다 하나에서 학습하고 다른 환경들에서 평가하면, source 환경의 선택에 따라 어떤 능력이 남는지 비교할 수 있다. 저자들은 그 차이를 state의 정보량과 계획 복잡성으로 분석하고, distractor를 추가하는 개입과 SFT datamix·thinking 조건을 통해 설명의 범위를 점검한다.

### 환경 4종 (§4.1, Table 1)

| 환경 | Task | State richness (평균 글자 수) | Planning complexity (평균 trajectory 길이) | Cross-domain robustness |
|---|---|---|---|---|
| **WebShop** | 웹에서 조건에 맞는 상품 찾아 구매 | High (3063) | Low (33.9) | Medium |
| **Sokoban** | 격자 퍼즐: 박스 하나를 target까지 밀기 | High (3114) | High (44.0) | High |
| **ALFWorld** | 여러 방·물체가 있는 가정 내 task | Low (1572) | Medium (42.0) | Low |
| **SciWorld** | ALFWorld와 비슷한 조작을 실험실에서. action space가 더 크고 다단계 실험 | Medium (2851) | High (43.5) | High |

- Sokoban은 원래의 행렬 표현 대신 벽·플레이어·박스·target을 **좌표 목록 텍스트**로 표현했다. 행렬 표현으로는 Llama-8B-Instruct의 RL이 의미 있게 오르지 않았기 때문이다(§4.1).

### 초기 정책 두 개 (§4.2)

Base는 `Llama-3.1-8B-Instruct`다. OOD 변화를 보려면 초기 정책이 모든 평가 도메인에서 0이 아닌 성능을 내야 하는데, base 모델은 WebShop과 SciWorld에서 거의 0이다.

- **`Ckpt V1`**: WebShop에서 20 step RL을 돌린 것. WebShop ID 성공률 34.4%. 이후 모든 실험의 출발점이 된다.
- **`Ckpt V2`**: SciWorld는 RL을 오래 돌려도 거의 0이라 **SFT warmup(mid-training)**을 했다. SciWorld expert 데이터 + `Ckpt V1`이 생성한 ALFWorld/WebShop trajectory를 **19:3:1** 비율(state-action pair 수 기준)로 섞었다. ALFWorld/WebShop 데이터는 성공 여부로 필터링하지 않았다. LlamaFactory로 100 step 학습했다(부록 A.5).
- V2는 Sokoban 데이터를 SFT에 포함하지 않았다. 이 점이 §7.1 분석의 핵심 변수가 된다.

### 학습·평가 프로토콜 (§4.1, 부록 A)

- 한 도메인에서 RL을 하고 나머지 셋에서 평가한다(single-source transfer). Ckpt V1에서는 SciWorld가 거의 0이라 평가에서 제외했다.
- 150 step 학습, 15 step마다 checkpoint를 저장한다. **마지막 4개 checkpoint(105–150 step)의 pass@1 평균**을 보고하고, seed 3개로 반복했다.
- 알고리즘은 **vanilla GRPO**의 multi-turn 확장이다(verl-agent 사용). 공통 설정: group size 8, step당 task prompt 16개, lr 1e-6, KL 계수 0.01, 성공 시 reward 10 / 실패 시 0, invalid action −0.1, rollout temperature 1.0, validation temperature 0.4. 최대 step은 ALFWorld/SciWorld 50, WebShop/Sokoban 15. GPU는 A100 8장.
- 기본 설정에서는 매 step `<think>…</think><action>…</action>` 형식으로 **step-by-step reasoning을 켠다**(부록 B 예시).

### 정의와 수식

**OOD Ranking Score**(↓, §5): 각 OOD 평가 도메인마다 학습 도메인들을 성공률 순으로 순위 매기고(1 = 최고), 그 순위를 합한 값이다. 성공률 차이가 0.5 미만이면 동순위로 처리한다(Table 8).

**State information richness**(§5): 수집한 trajectory의 state 평균 **글자 수**. Ckpt V2로 도메인당 128개 trajectory를 수집했다($T_{\max}=50$). 네 도메인 모두 state가 압축적으로 쓰여 있으니 글자 수를 정보량의 근사로 쓸 수 있다는 논리다.

**Planning complexity**(§5): 같은 128개 trajectory의 **평균 길이**. 실패한 trajectory는 $T_{\max}=50$으로 계산한다. 그래서 이 값에는 task horizon과 goal reachability가 함께 반영된다.

**OOD Change**(§6): augmentation 유무에 따른 OOD 성공률 합의 상대 변화.

$$\Delta_{\text{OOD}} = \frac{\sum_{e\in\mathcal E^{\text{eval}}}\big(G(\pi_{\text{w/ Aug}},e)-G(\pi_{\text{w/o Aug}},e)\big)}{\sum_{e\in\mathcal E^{\text{eval}}}G(\pi_{\text{w/o Aug}},e)}\times 100\%$$

**State Information Augmentation**(§6): 원래 state $s$에 goal과 무관한 텍스트 $\delta$를 분량 $\epsilon$만큼 끼워 넣어 $s'=\mathrm{Augment}(s,\delta,\epsilon)$를 만든다. **agent 입력에만** 적용하고 transition과 reward는 원래 state 기준으로 돌아간다.

## 실험과 핵심 결과

### 실험 설정

| 축 | 내용 |
|---|---|
| 모델 | Llama-3.1-8B-Instruct (한 종류) |
| 초기 정책 | Ckpt V1 (WebShop 20-step RL), Ckpt V2 (V1 + SFT warmup) |
| 학습 도메인 | V1: ALFWorld, WebShop, Sokoban / V2: 여기에 SciWorld 추가 |
| 지표 | OOD 성공률, ΔID, OOD Ranking Score, ΔOOD |
| 개입 | state augmentation, SFT warmup 유무(V1 vs V2), thinking on/off, 학습 step 연장 |

### Finding 1. RL은 OOD 성능을 깎는다. 깎이는 정도는 학습 도메인에 따라 다르다 (Table 2)

| 초기 정책 | 학습 도메인 | ALFWorld | WebShop | Sokoban | SciWorld |
|---|---|---|---|---|---|
| V1 | *Base* | 25.8 | 34.4 | 12.5 | 3.1 |
| V1 | ALFWorld | ΔID=+47.2 | 15.2 | 14.0 | – |
| V1 | WebShop | 10.8 | ΔID=+23.6 | 12.8 | – |
| V1 | Sokoban | 20.5 | 15.0 | ΔID=+36.0 | – |
| V2 | *Base* | 21.9 | 38.3 | 14.1 | 18.0 |
| V2 | ALFWorld | ΔID=+60.6 | 30.5 | 9.8 | 10.0 |
| V2 | WebShop | 17.0 | ΔID=+28.7 | 9.0 | 13.8 |
| V2 | Sokoban | 20.0 | 34.0 | ΔID=+35.7 | 13.0 |
| V2 | SciWorld | 19.8 | 35.8 | 12.0 | ΔID=+20.2 |

- ID는 크게 오르는 반면 OOD는 대부분 base보다 낮다. 예외는 V1에서 ALFWorld/WebShop으로 학습했을 때의 Sokoban(14.0, 12.8 vs base 12.5)뿐이다.
- OOD Ranking Score(Table 3)는 다음과 같다.
  - **V2 기준**: SciWorld 3(1위) < Sokoban 5 < WebShop 6 < ALFWorld 8.
  - **V1+V2 합산**(SciWorld 제외): Sokoban 7(1위) < WebShop 10 = ALFWorld 10.

### Finding 2. State information richness × planning complexity가 둘 다 높아야 일반화를 잘 보존한다 (Table 3, §5)

- 상위권인 Sokoban(3114자, 44.0 step)과 SciWorld(2851자, 43.5 step)는 **두 축이 모두 높다.**
- 하위권은 적어도 한 축이 낮다.
  - **WebShop**: state는 길다(상품 설명, feature 목록). 하지만 planning은 얕다. 상품을 훑으며 속성을 맞추면 끝난다.
  - **ALFWorld**: task 논리는 복잡하지만 state가 짧고 단순하다(1572자).
- **Realism은 설명력이 없다.** 추상 grid 퍼즐인 Sokoban은 현실적인 SciWorld로도 잘 전이되는 반면(V2: Sokoban→SciWorld 13.0 vs ALFWorld→SciWorld 10.0), 현실적인 가정 환경인 ALFWorld는 최하위다.
- 저자들도 도메인이 4개뿐이므로 이 결과는 **"indicative rather than conclusive"**이며, 특성화가 다소 ad hoc하고 인과를 분리하지 못한다고 인정한다(§5 말미).

### Finding 3. "ID를 많이 배울수록 OOD를 잃는다"만으로는 설명되지 않는다 (Table 4, Table 9)

- Table 2만 보면 ΔID가 큰 쪽이 OOD를 더 잃는 단순한 trade-off처럼 보인다. 이를 확인하려고 Sokoban은 300 step, SciWorld는 240 step까지 학습을 연장했다.
- Sokoban의 ΔID는 +67.9(V1) / +68.4(V2)까지 올라 ALFWorld·WebShop(150 step)보다 커졌다. 그래도 OOD Ranking Score는 **8(1위, V1+V2) / 6(2위, V2)**로 순위가 유지됐다. SciWorld(ΔID +41.8)도 V2에서 1위(점수 5)를 유지했다.
- 다만 절대 OOD 성공률은 학습을 연장하면서 계속 내려간다. 예: SciWorld 학습의 OOD는 150 step 19.8/35.8/12.0에서 240 step 16.9/33.6/11.8로 낮아졌다(Table 9). 유지된 것은 *상대 순위*다.

### Finding 4. State Information Augmentation이 OOD 성능을 일관되게 높인다 (Table 5, 부록 C)

도메인별 augmentation 방식은 다음과 같다(부록 C).
- **ALFWorld**: 가짜 distractor 물체 문장을 넣는다. 예: "Near the sink, there is a *mug 3* that is *covered in dust*". 물체 type 28종 × 묘사 19종 템플릿, 약 12 token짜리 문장을 ⌊ε/12⌋개 무작위 위치에 삽입한다.
- **WebShop**: 광고 문장(`[AD] 30% off — shop cozy cardigans today!`), 선택할 수 없는 가짜 상품(`Warning: out of stock hoodie (cannot be selected)`), 상세 페이지의 사소한 feature를 넣는다.
- **Sokoban**: 격자 밖 좌표에 "(r, c) shows a Broken crate (beyond the wall; unreachable)." 같은 줄을 ⌊ε/10⌋개 추가한다.
- 도메인 지식이 거의 없을 때의 대안도 제시한다. 다른 도메인의 observation을 끼워 넣는 방식이다. 예: ALFWorld state에 "there is a monitor showing a shopping webpage: <WebShop state>"를 추가한다.
- ε가 너무 작으면 효과가 없고, 너무 크면 ID 학습을 방해한다. ID가 눈에 띄게 떨어지기 직전까지 ε를 올리라고 권한다. V2에서는 trajectory마다 **50% 확률**로만 적용해 ID 하락을 막았다(Table 11: V1은 ε=300/30/80, 확률 100% / V2는 ε=200/40/80, 확률 50%).

| 초기 정책 | 학습 도메인 | OOD (기본 → +Aug) | ΔID (기본 → +Aug) | ΔOOD (원문) |
|---|---|---|---|---|
| V1 | ALFWorld | WebShop 15.2→23.8, Sokoban 14.0→11.2 | 47.2→47.7 | +32.6% |
| V1 | WebShop | ALFWorld 10.8→17.8, Sokoban 12.8→11.0 | 23.6→27.6 | +35.5% |
| V1 | Sokoban | ALFWorld 20.5→21.2, WebShop 15.0→24.8 | 36.0→36.7 | +42.5% |
| V2 | ALFWorld | 30.5→30.3, 9.8→11.0, 10.0→12.5 | 60.6→56.6 | +7.0% |
| V2 | WebShop | 17.0→25.8, 9.0→11.8, 13.8→15.5 | 28.7→27.7 | +33.4% |
| V2 | Sokoban | 20.0→20.8, 34.0→37.0, 13.0→13.0 | 35.7→37.4 | +5.7% |

- 6개 설정 모두 ΔOOD가 양수이고, ID는 대체로 유지된다. 저자들은 이를 state richness가 robustness의 결정 요인이라는 가설에 대한 **"causal validation"**으로 해석한다(§6).
- 저자들은 state randomization 기법 자체보다 **가설 검증 도구**라는 점이 기여라고 강조한다(§2.3). 그러면서도 semantic textual noise와 distraction 강도 조절이 agentic RL의 OOD 하락 방지에 도움이 된다는 것을 처음 보였다고 주장한다.

### Finding 5. SFT warmup은 양날의 검이다. 포함된 도메인은 지키고, 빠진 도메인은 더 잊게 한다 (Table 6, §7.1)

같은 RL을 V1(SFT 없음)과 V2(SFT 있음)에서 시작했을 때, 평가 도메인 성공률이 각 초기 정책 대비 얼마나 변했는지 비교한다.

| RL 학습 | 평가 | V2 SFT에 포함? | V1 상대변화 | V2 상대변화 |
|---|---|---|---|---|
| Sokoban | WebShop | Yes | −56.4% | **−11.2%** |
| Sokoban | ALFWorld | Yes | −20.5% | **−8.7%** |
| ALFWorld | WebShop | Yes | −68.6% | **−53.5%** |
| ALFWorld | Sokoban | **No** | **+12.0%** | −30.5% |
| WebShop | ALFWorld | Yes | −58.1% | **−22.4%** |
| WebShop | Sokoban | **No** | **+2.4%** | −36.2% |

- SFT datamix에 **포함된 도메인**에서는 RL 중 OOD 하락이 일관되게 훨씬 작다. 성공 여부로 거르지 않은 self-generated trajectory만으로도 이 효과가 났다. 저자들은 "환경 dynamics에 노출만 시켜도 도메인 지식이 consolidate된다"고 해석한다.
- **빠진 도메인(Sokoban)**은 반대다. V1에서는 오히려 소폭 오르는데 V2에서는 크게 떨어진다.
- 저자들의 권고: 배포 도메인을 모르면 mid-training 강도를 조절하고, datamix를 가능한 한 넓게 잡을 것.

### Finding 6. Thinking을 끄면 ID는 비슷해도 OOD가 무너진다 (Table 7, §7.2)

| 초기 정책 | 학습 도메인 | ΔID (think → no-think) | 대표 OOD (think → no-think) | OOD Change (원문) |
|---|---|---|---|---|
| V1 | ALFWorld | 47.2 → 51.2 | WebShop 15.2 → **0.8** | −119.0% |
| V1 | WebShop | 23.6 → 27.6 | Sokoban 12.8 → 17.8 | +16.9% |
| V1 | Sokoban | 36.0 → 23.0 | ALFWorld 20.5 → 16.0 | −9.1% |
| V2 | ALFWorld | 60.6 → 51.3 | WebShop 30.5 → **1.0** | −234.2% |
| V2 | WebShop | 28.7 → 14.5 | ALFWorld 17.0 → 4.5 | −82.5% |
| V2 | Sokoban | 35.7 → 60.7 | ALFWorld 20.0 → 9.2 | −216.5% |
| V2 | SciWorld | 20.2 → 17.5 | ALFWorld 19.8 → 6.2 | −216.9% |

- V2에서는 ALFWorld/Sokoban/SciWorld 학습 모두 OOD Change가 −200%를 넘는다.
- ID는 thinking 없이도 비슷하거나 더 높을 수 있다. V1 ALFWorld의 ID는 73.0% → 77.0%, V2 Sokoban은 ΔID 35.7 → 60.7이다.
- 저자들의 해석: reactive policy는 학습 환경에 맞춘 도메인 특화 패턴만으로도 ID 최적 정책을 찾을 수 있지만, 그런 heuristic은 깨지기 쉽고 전이되지 않는다. reasoning을 명시하게 하면 얕은 heuristic에 overfit하는 것을 막는다. DQN처럼 reasoning 없이 사람 수준에 이르는 reactive 정책과 대비하며, **explicit reasoning이 전이에 필수**라는 관점을 제시한다.

## 한계와 해석

### 저자들이 밝힌 한계

- 도메인이 4개뿐이라 Finding 2는 "indicative rather than conclusive"이고, 특성화가 ad hoc하며 인과를 완전히 분리하지 못한다.
- 더 많은 도메인과 다양한 모델 규모로 확장해 보편성을 확인해야 하고, 추가 요인이 있을 수 있다.
- 두 지표(글자 수, trajectory 길이)는 경험적 proxy다. 더 형식적이고 인과적으로 근거 있는 정의가 필요하다.
- 향후 과제: 자동 environment augmentation, reasoning 구조와 일반화의 관계, 지식 consolidation과 유연성을 동적으로 조절하는 adaptive SFT warmup, 이론적 해석.

### AI의 해석

<aside class="note-callout" markdown="1">
**아래는 AI 정리자의 해석이며 원논문의 주장과 구분합니다.**
</aside>

> 이하는 논문의 주장이 아니라 정리자의 해석과 검산이다.

1. **"Planning complexity" 지표는 사실상 "초기 정책의 실패율"이다.** 실패한 trajectory를 $T_{\max}=50$으로 세면 평균 길이 ≈ $(1-p)\cdot 50 + p\cdot L_{\text{succ}}$가 된다. Table 2의 V2 base 성공률 $p$를 넣어 역산하면 다음과 같다.
   - Sokoban (p=14.1%): $(1-p)\cdot50$ = 43.0 → 성공 trajectory 평균 길이 ≈ 7.4
   - WebShop (p=38.3%): 30.9 → ≈ 8.0
   - ALFWorld (p=21.9%): 39.1 → ≈ 13.5
   - SciWorld (p=18.0%): 41.0 → ≈ 13.9

   성공 trajectory 자체는 Sokoban이 가장 짧은 편이다. 44.0이라는 "high complexity"는 거의 전부 낮은 성공률에서 나온다. 저자들도 이 지표가 goal reachability를 반영한다고 밝히지만, 그렇다면 축의 이름은 "planning complexity"보다 **"초기 난이도(1 − base success)"**에 가깝다. 또 WebShop/Sokoban은 학습 시 최대 step이 15인데 지표는 $T_{\max}=50$으로 쟀다(부록 A). 측정 조건과 학습 조건이 다르다.
2. **n=4이고, 순위 차이가 작다.** Ranking Score의 차이는 순위 1–2칸 수준이다(V2: 3/5/6/8). 동순위 기준이 0.5%p인데 seed 3개의 분산(std)은 보고되지 않았다. 예: V2 ALFWorld 평가에서 Sokoban 20.0 vs SciWorld 19.8은 동순위, WebShop 17.0은 2위다(dense ranking). V1만 보면 ALFWorld와 Sokoban이 순위 합 2로 **동률**이다. 합산 결과에서 Sokoban이 1위인 것은 V2 덕분이다. 지표 2개 × 도메인 4개에서 "둘 다 높으면 좋다"를 읽어내는 것은 사후 해석의 여지가 크다.
3. **Augmentation이 "richness" 때문에 효과가 있는지는 분리되지 않았다.** 무작위 텍스트 삽입은 일반적인 input perturbation/data augmentation(정규화, 입력 다양성 증가)으로도 작동할 수 있다. 같은 분량이지만 "처리할 필요가 없는" 텍스트(예: 고정된 boilerplate)를 넣는 대조군이나, 다양성만 늘리고 길이는 그대로 두는 대조군이 있어야 "information richness → 일반화"라는 인과 해석이 강해진다. 개별 셀을 보면 떨어진 곳도 있다(V1 ALFWorld→Sokoban 14.0→11.2, V1 WebShop→Sokoban 12.8→11.0). 효과는 합계 기준이다.
4. **표 수치의 내부 불일치** (검산 결과)
   - **Table 5의 V1 ΔOOD**: 본문 공식대로 계산하면 ALFWorld +19.9%, WebShop +22.0%, Sokoban +29.6%다. 표에 적힌 +32.6/+35.5/+42.5%와 다르다(V2의 +7.0/+33.4/+5.7%는 공식과 일치). 본문 §6은 V1 ALFWorld를 "+7.0%"라고 쓰는데 이것도 표와 다르다. 방향(모두 양수)은 같지만 크기는 과장돼 있을 수 있다.
   - **Table 7의 OOD Change**: 정의가 명시되어 있지 않다. −234% 같은 값은 성공률 합의 상대 변화로는 나올 수 없다(하한이 −100%). 정리자가 시도한 두 정의(합의 상대 변화, 도메인별 상대 변화의 합)로는 재현되지 않았다. 또 V2 ALFWorld w/o Thinking 행은 SciWorld 값이 비어 있다. 방향성(대부분 큰 하락)은 원 수치에서도 분명하지만, 퍼센트 값 자체는 인용에 주의해야 한다.
   - **Table 6의 ALFWorld→WebShop 행**: V1 10.8 / V2 17.8로 적혀 있지만, Table 2에서는 각각 15.2 / 30.5다. 10.8과 17.0은 WebShop→ALFWorld 값이라 옮겨 적다가 섞인 것으로 보인다. Table 2 값으로 다시 계산하면 V1 −55.8%, V2 −20.4%이고, 결론(SFT에 포함된 도메인은 덜 잊는다)은 오히려 더 강해진다.
   - **Table 4 vs Table 10**: 연장 학습의 ΔID가 본문 표(67.9/68.4/41.8)와 부록 순위 표(64.5/66.5/40.7)에서 다르다. 부록 표 두 개는 같은 label을 쓴다.
   - 초록은 "text-level similarity"도 주요 요인이 아니라고 하지만, 본문에서 텍스트 유사도를 측정한 분석은 찾지 못했다. realism만 정성적으로 다룬다.
5. **SFT 비교(V1 vs V2)에는 confound가 있다.** V2는 SFT 외에도 SciWorld expert 데이터가 19/23을 차지하고 출발 성공률도 다르다. "SFT warmup 일반"의 효과라기보다 *이 특정 datamix*의 효과다. Sokoban이 datamix에서 빠진 유일한 도메인이라, "빠진 도메인은 잊는다"는 결론의 근거는 Sokoban 2칸뿐이다.
6. **Thinking ablation이 다른 것까지 바꿨을 가능성**: w/o thinking은 출력 형식, 응답 길이, 토큰 예산도 함께 바뀐다. OOD 1.0%처럼 극단적인 붕괴는 정책 능력 상실이 아니라 **형식이나 action 파싱 실패**일 수도 있다(invalid action 비율이 보고되지 않음). 그래도 "ID는 같거나 높은데 OOD는 무너진다"는 패턴은 heuristic overfitting 가설과 잘 맞고, 이 논문에서 가장 인상적인 결과로 보인다.
7. **모델 하나(Llama-3.1-8B), 8B 규모, vanilla GRPO** 조건이다. 더 큰 모델이나 다른 RL 알고리즘(예: KL을 끈 설정)에서도 순위가 유지될지는 알 수 없다. RL 학습 step도 150으로 짧다.

> 💭 종합하면, 이 논문의 가장 견고한 메시지는 세 가지로 보인다. (i) agent RL에도 OOD 세금이 실재하며 학습 도메인에 따라 크기가 다르다. (ii) thinking을 끄면 OOD가 무너진다. (iii) mid-training datamix의 coverage가 forgetting을 좌우한다. "richness × complexity" 축은 흥미로운 가설이지만, 지표가 base 성공률과 겹치고 n=4라서 *환경 선택 지침*으로 쓰기에는 아직 약하다.

## 더 생각해볼 질문

아래는 AI가 정리 과정에서 제안한 후속 질문입니다. 본인이 직접 작성한 의견이나 논문에서 검증한 결과로 표시하지 않습니다.

- **난이도 vs 정보량 분리**: planning complexity가 사실상 "초기 실패율"이라면, Finding 2는 "어려운 환경(낮은 base success)에서 학습할수록 덜 잊는다"로 다시 쓸 수 있다. 같은 환경 안에서 task 난이도만 바꾸는 curriculum 실험으로 이것을 분리할 수 있을까?
- **Augmentation 대조군 설계**: 같은 길이의 *무의미한 고정 텍스트* vs *goal과 무관하지만 의미 있는 distractor* vs *다른 도메인 observation*. 이 셋 중 무엇이 OOD를 지키는지 보면 "richness"인지 "distractor 처리 능력"인지 갈린다.
- **[Agent Plasticity](/posts/agent-plasticity/)와의 연결**: 저쪽은 weight를 고정한 채 경험으로 개선하는 능력을 잰다. 이 논문의 "generalization tax"는 weight를 바꿨을 때 잃는 것이다. RL post-training이 *in-context plasticity*도 깎는지, 그리고 thinking/augmentation이 그것도 보존하는지 궁금하다.
- **Thinking의 역할**: OOD 보존이 reasoning 내용 때문인지, 단순히 출력 토큰이 늘어 정책 분포가 덜 날카로워지기(entropy 유지) 때문인지 구분할 수 있을까? 예: 내용 없는 filler thinking과 비교.
- **Mid-training datamix 설계**: "노출만 시켜도 consolidate된다"면, 배포 후보 도메인의 *성공 여부를 거르지 않은* self-rollout을 소량 섞는 것만으로 forgetting 보험이 될 수 있다. 비용 대비 효과가 큰 레시피로 보인다.
- [LoGRA](/posts/logra-gradient-sketches/)의 KL budget처럼 정책 변화량을 직접 제한하면 generalization tax도 줄어들까? 이 논문은 KL 계수를 0.01로 고정했다.

## 참고자료

- [원논문](https://arxiv.org/abs/2601.18217)
- [논문 PDF](https://arxiv.org/pdf/2601.18217)

<details markdown="1">
<summary>용어 정리</summary>

| 용어 | 뜻 |
|---|---|
| Generalization tax | 특정 도메인 RL로 ID가 오르는 대신 학습하지 않은 도메인에서 잃는 성능 |
| In-domain (ID) validation | 학습 환경과 같은 환경의 held-out task로 평가 |
| Cross-domain / OOD generalization | 학습에 쓰지 않은 다른 환경(도구·state·dynamics가 다름)으로 평가 |
| State information richness | state에서 agent가 처리해야 할 정보량. 평균 글자 수로 근사 |
| Planning complexity | 긴 다단계 계획이 필요한 정도. 실패를 $T_{\max}$로 센 평균 trajectory 길이로 근사 |
| OOD Ranking Score | OOD 평가 도메인별 순위의 합. 낮을수록 좋다 |
| ΔID | RL 후 ID 성공률 − 초기 정책 ID 성공률 |
| ΔOOD (OOD Change) | augmentation 유무에 따른 OOD 성공률 합의 상대 변화 |
| State Information Augmentation | goal과 무관한 distractor 텍스트를 state에 삽입. transition과 reward는 그대로 |
| ε (information volume) | 삽입하는 distractor의 분량 |
| SFT warmup / mid-training | RL 전에 SFT로 도메인 지식을 주입하는 단계. 여기서는 Ckpt V2 |
| Ckpt V1 / V2 | WebShop 20-step RL 체크포인트 / V1 + SFT warmup 체크포인트 |
| Reactive policy | reasoning 없이 state에서 바로 action을 출력하는 정책(w/o Thinking) |
| GRPO | group 내 reward를 정규화해 advantage를 계산하는 critic-free RL |

</details>
