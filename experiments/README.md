# Mục lục `experiments/`
82 script. Mỗi script có docstring đầy đủ ở đầu file — dòng dưới đây chỉ là tóm tắt.
> **Trước khi đọc bất cứ file nào có chữ `rq<số>` trong tên**: xem bảng dịch số RQ
> trong `README.md` gốc. Repo có ba thế hệ đánh số RQ và chúng **không khớp nhau**.

## 1. THU THAP DU LIEU — chay tren he thong THAT dang chay
*Can Sock Shop (hoac he dich) dang len. Khong chay duoc offline.*
- **`load_sweep_collect.py`** — QUET TAI CO KIEM SOAT SOCK SHOP + THU THAP TELEMETRY
- **`probe_feature_chain.py`** — DO CHUOI GOI THUC TE CUA TINH NANG (chuc nang + backend nao bi goi, moi lan dung goi may lan)
- **`data_contract_check.py`** — KIEM TRA HOP DONG DU LIEU (docs/DATA_FRAMEWORK.md muc 4 va 6)
- **`forecast_traces.py`** — BO SINH CHUOI TAI CHO BAI TOAN FORECASTING (thuan numpy, khong can Docker)
- **`download_trainticket_data.py`** — DOWNLOAD TRAIN TICKET TELEMETRY DATA FROM HUGGING FACE
- **`download_onlineboutique_data.py`** — DOWNLOAD ONLINE BOUTIQUE TELEMETRY DATA FROM HUGGING FACE

## 2. NHANH KHA THI (RQ5) — dong bang truoc, danh gia sau
*Day la duong di chinh cua RQ5. THU TU QUAN TRONG: freeze_* phai chay TRUOC khi do ramp, neu khong thi khong con la du doan tien dang ky.*
- **`freeze_predictions.py`** — DONG BANG DU DOAN (pre-registration) TRUOC KHI DO DAP AN
- **`freeze_p3_prospective.py`** — DONG BANG P3 TIEN CUU (sau khi CO code, TRUOC khi chay ramp/xem dap an)
- **`fit_feature_costs.py`** — HIEU CHINH THAM SO P2 (chi phi moi lan goi c_s va chi phi gateway x) -- CHI tap PHAT TRIEN (promo, recs)
- **`evaluate_frozen.py`** — DANH GIA BAN DONG BANG SO VOI DAP AN (Pha B)
- **`evaluate_p3.py`** — P3 -- P2 + BOI SO GOI (k_s) DO DUOC, THAY VI GIA DINH k_s=1 (CHUAN DOAN HOI CUU / RETROSPECTIVE)
- **`decompose_errors.py`** — PHAN TACH NGUON SAI SO CUA P1 (chi tap PHAT TRIEN: base/promo/recs)
- **`control_chain_distribution.py`** — DOI CHUNG CHAIN-SAI DUOI DANG PHAN PHOI, thay vi MOT lan boc duy nhat

## 3. SO SANH MO HINH & THONG KE (RQ3 Measurability)
*Sinh 05_model_comparison.csv va cac kiem dinh kem hieu chinh Holm.*
- **`model_comparison.py`** — Model Comparison: SCM vs Linear Regression vs Gradient Boosting vs Gaussian Process
- **`rq2_statistical_analysis.py`** — RQ2 statistical analysis, corrected
- **`statistical_rigor.py`** — HIEU CHINH FAMILY-WISE + KHOANG TIN CAY cho bo kiem dinh da mo rong
- **`evaluation_suite.py`** — HỆ THỐNG ĐÁNH GIÁ TỔNG HỢP (EVALUATION SUITE) CHO SCM
- **`scm_pipeline.py`** — SCM Impact Prediction Pipeline - Structured Evaluation
- **`compare_with_ground_truth.py`** — Direct Ground-Truth Matching with RE2-SS Raw Data (FULL AGGREGATED VERSION)
- **`cpu_mape_theoretical_floor.py`** — "TRAN LY THUYET" cua CPU MAPE: neu dung ham workload->CPU TOT NHAT CO THE
- **`forecast_data_audit.py`** — KIEM DINH DU LIEU THEO CHUAN FORECASTING

## 4. PARSER & SCOPE GATE (RQ1 Necessity, RQ2 Guarantee)
*Benchmark parser 50 prompt, cac lop guard, va hieu chinh conformal cua scope gate.*
- **`parser_benchmark_suite.py`** — Parser Benchmark Suite for Q1 Academic Paper (RQ3 Validation)
- **`g1_g3_adversarial_stress_test.py`** — G1 & G3 Adversarial Stress Test — Can a Crafted Prompt Actually Trip These Guards?
- **`g3_independent_verification.py`** — G3 Independent Verification — Does the Clamp Do Real Work Without Prompt Self-Constraint?
- **`g6_scope_gate_adversarial_test.py`** — Scope Gate (Layer 1) Adversarial Stress Test — Can an Evasive Prompt Sneak Past Refusal?
- **`g6_scope_gate_hitl_test.py`** — Scope Gate HITL Escalation — Full Validation
- **`g6_scope_gate_layerA_test.py`** — Scope Gate Redesign, Full Validation — Layer A (self-declared field) + Layer B (backstop)
- **`g7_ood_guard_test.py`** — G7 (Proposed Guard) — Does an OOD-Confidence Check Add Value Within G2's Legal Range?
- **`g_threshold_sensitivity.py`** — Threshold Sensitivity Analysis for G2/G3/G5 — Are the Specific Numbers Load-Bearing?
- **`scope_gate_embedding_calibration.py`** — Scope Gate Redesign — Calibrating a Semantic-Embedding Threshold on Real Data
- **`scope_gate_conformal_calibration.py`** — Conformal Calibration for the Scope Gate (Layer 1) — a Real Trial
- **`scope_gate_holdout_test.py`** — Scope Gate: genuine held-out test on the conformal-calibrated Scope Gate
- **`rq3_rerun_new_scope_gate.py`** — RQ3 partial re-run on the new (sole) conformal Scope Gate architecture
- **`rq5_coordination_overhead.py`** — RQ5 — Multi-Agent Coordination Overhead vs. Single-LLM-Call Baseline

## 5. CHON CANH SCM & CO CHE (RQ4 Attribution)
*Chon canh phai hoc (backpressure, latency backprop) va so sanh lop co che.*
- **`select_scm_edges.py`** — TIEN TINH TOAN canh SCM (Tier 2.5 "backpressure") TU DONG, TU LOGS
- **`scm_mechanism_bakeoff.py`** — BAKE-OFF CHON CO CHE DU PHONG CHO SCM (Alibaba v2021)
- **`scm_builder_final_measurement.py`** — DO LUONG CHOT: engine xay SCM tong quat (template + held-out + cache)
- **`generalized_covariate_selection.py`** — TONG QUAT HOA: tu dong chon covariate bo sung (khong con chan doan tay tung node)
- **`replace_vs_add_edge_test.py`** — KIEM DINH: "Thay canh" (shipping_workload -> orders_cpu) hay "Them canh" (giu ca 2)?
- **`backpressure_edge_accuracy_test.py`** — KIEM DINH: Them canh backpressure (caller_cpu lam parent thu 2) co thuc su
- **`backpressure_edge_ood_safety_test.py`** — AN TOAN NGOAI SUY: canh backpressure moi them vao capacity_agent.py::train_accurate_path
- **`memsocket_edge_accuracy_test.py`** — KIEM DINH held-out (OOD Gold Standard, giong RQ1/backpressure_edge_accuracy_test.py)
- **`tt_backpressure_edge_accuracy_test.py`** — TRAIN TICKET: kiem dinh canh backpressure (caller_cpu lam parent thu 2) tren
- **`tt_backpressure_edge_ood_safety_test.py`** — TRAIN TICKET: an toan ngoai suy khi them 50 canh backpressure (gain>0.03) vao
- **`tt_backpressure_multiparent_accuracy_test.py`** — SUA MOT KHOANG TRONG KIEM DINH: tt_backpressure_edge_accuracy_test.py danh gia
- **`causil_latency_edge_trial.py`** — CAUSIL LATENCY-BACKPROPAGATION EDGE -- FEASIBILITY TRIAL
- **`causil_latency_edge_ood_safety_trial.py`** — CAUSIL LATENCY EDGE -- PHA 4: OOD-SAFETY VALIDATION
- **`nonlinear_mechanism_trial.py`** — THU NGHIEM: Co che hoi quy (mechanism) nao chinh xac hon trong DAI TAI THUC TE?
- **`saturating_mechanism_trial.py`** — THU NGHIEM TIEP THEO (sau nonlinear_mechanism_trial.py): mo hinh BAO HOA CO THAM SO
- **`log_transform_trial.py`** — THU NGHIEM CHAN DOAN: log-transform co sua duoc 4 node lech nang khong?
- **`mechanism_decision_quality_trial.py`** — THU NGHIEM 2: Doi mechanism co thuc su cai thien CHAT LUONG QUYET DINH khong?

## 6. BO DU LIEU KHAC — Alibaba, Train Ticket, Online Boutique
*Dung de doi chung ngoai Sock Shop.*
- **`alibaba_signal_check.py`** — Kiem tra Alibaba v2021 co chua tin hieu workload -> resource khong
- **`alibaba_confounder_test.py`** — Alibaba co confounder ha tang khong? -- phep kiem co the LAT ket luan ve do()
- **`alibaba_forecast_eval.py`** — Danh gia du bao capacity tren Alibaba v2021 -- CO CHE DO THAT BAI
- **`alibaba_latency_queueing_test.py`** — QUEUEING FORM CO XUNG DANG CHO LATENCY KHONG? (Alibaba v2021)
- **`alibaba_node_load_tier.py`** — Tang 1.5: tai node co cai thien du bao resource khong?
- **`alibaba_tier1_propagation.py`** — Lan truyen Tier-1 tren Alibaba: do thi co gia tri gi hon gia dinh delta deu?
- **`alibaba_workload_per_instance_test.py`** — BIEN HIEP BIEN NAO CHO LATENCY: WORKLOAD TONG, WORKLOAD/INSTANCE, HAY SO INSTANCE?
- **`trainticket_evaluation.py`** — ĐÁNH GIÁ SCM TRÊN HỆ THỐNG THỨ HAI: TRAIN TICKET (RQ1/RQ2, PER-SYSTEM)
- **`elasticity_transfer_loso.py`** — LEAVE-ONE-SYSTEM-OUT: DO CO GIAN (ELASTICITY) CO CHUYEN GIAO GIUA CAC HE THONG?
- **`baro_on_real_rcaeval_scenarios.py`** — BARO on RCAEval's real Sock Shop fault-injection scenarios (RE2-SS)

## 7. CALL CHAIN / TAXONOMY
*Kiem chung chain khai thac tu log va cac chan doan lien quan.*
- **`callchain_from_logs_validation.py`** — KIEM CHUNG: callchain_from_logs.py mine dung services toi muc nao?
- **`call_chain_neighbor_diagnostic.py`** — CHAN DOAN: Yeu to ngoai Workload nao giup giai thich phan du CPU (R2 am)?

## 8. HE DANH SO CU (RQ6–RQ12) — GIU DE TAI LAP, KHONG CON DUOC THAM CHIEU
*CANH BAO: so RQ trong ten cac file nay thuoc khung CU (RQ1–RQ12), KHONG phai khung 5 RQ hien tai cua bai bao. Xem bang dich trong README.md goc. Giu lai vi chung la nguon goc cua cac ket qua am da bao cao (backpressure, tier decomposition, backdoor adjustment).*
- **`rq6_attribution_validity.py`** — RQ6 — Does Interventional Shapley Attribution Correctly Localize Impact?
- **`rq6_baro_comparison.py`** — RQ6 diagnostic follow-up #4 — Does BARO (FSE'24), a non-causal statistical
- **`rq6_hop1_node_diagnostic.py`** — RQ6 diagnostic follow-up #2 — Why do ts-order-service_cpu and
- **`rq6_hop_distance_diagnostic.py`** — RQ6 diagnostic follow-up — Does tier1_driven attribution accuracy degrade
- **`rq6_propagation_path_diagnostic.py`** — RQ6 diagnostic follow-up #3 — Does the TRUE PROPAGATED signal (how much a
- **`rq6_tier_decomposition.py`** — RQ6 (Part B) — Does Shapley Attribution Correctly Split Tier 1 vs Tier 2?
- **`rq6_tier_decomposition_v2.py`** — RQ6 (Part B, v2) — Corrected Tier-1 vs Tier-2 Shapley Decomposition
- **`rq6_topology_check.py`** — RQ6 (Part A.3) — Does Shapley Attribution Respect Graph Topology?
- **`rq7_interventional_validity.py`** — RQ7: Fault injection lam GROUND TRUTH cho mot CAN THIEP thuc su
- **`rq7b_third_party_intervention.py`** — Bien the sach: fault tiem vao service THU BA, ca B va C deu KHONG bi tiem.
- **`rq7c_extrapolation_vs_noncausal.py`** — Phan dinh hai giai thich canh tranh cho viec R_A -> R_B vo duoi can thiep:
- **`rq8_confounder_evidence.py`** — Fix 3 -- Co ton tai confounder ha tang dung chung khong?
- **`rq9_backdoor_adjustment.py`** — Fix 2 -- Toan tu do() co noi dung khi can thiep tai NODE NOI BO khong?
- **`rq10_simultaneous_intervention.py`** — Huong 2 -- Can thiep DONG THOI len to tien va hau due: cho do() co noi dung
- **`rq11_counterfactual_validity.py`** — Huong phan thuc (Pearl tang 3) -- co kha thi khong?
- **`rq12_prospective_attribution.py`** — !!! THI NGHIEM NAY KHONG HOP LE -- GIU LAI DE GHI NHAN, KHONG DUOC TRICH DAN !!!
- **`rq_joint_gateway_gap.py`** — RQ MOI (thay the C_5b da bi bac): "gap" giua CAN THIEP DONG THOI va CONG DON LE
- **`future_rca.py`** — FUTURE RCA ENGINE — Pre-mortem Root Cause Analysis via SCM do-calculus
- **`latency_covariate_diagnostic.py`** — CHAN DOAN: them covariate nao giai thich duoc carts_latency-50/orders_latency-50?
