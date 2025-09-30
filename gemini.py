import pandas as pd
from ortools.linear_solver import pywraplp
import numpy as np
import os
from collections import defaultdict

# --- 1. 환경 설정 및 데이터 로드 ---

INPUT_FILE = "학급반편성CSP 문제 입력파일.csv"
OUTPUT_FILE = "학급반편성_최종결과_MIP.csv"

# 학급 정보
NUM_STUDENTS = 200
NUM_CLASSES = 6
CLASS_CAPACITIES = {
    1: 33, 2: 33, 3: 33, 4: 33,
    5: 34, 6: 34
}
CLASS_NAMES = list(CLASS_CAPACITIES.keys())

# --- 페널티 가중치 설정 (Soft Constraint 우선순위) ---
WEIGHTS = {
    # Hard Constraint를 대체하는 Soft Constraint (위반 절대 회피)
    'Bad_Relation': 20000,     # C: 나쁜 관계 배정 시 최고 페널티
    'Leader_Violation': 15000, # E: 리더십 최소 1명 미달 시 고 페널티
    'Mentor_Separation': 10000,# D: 멘토-비등교 쌍 분리 시 고 페널티
    'Capacity_Deviation': 500, # B: 정원 33/34명 편차 페널티

    # 기타 Soft Constraint (균형)
    'Prev_Class_Overlap': 500, # I: 전년도 학급 중복
    'Non_Attendance_Balance': 200, # G: 비등교 균등
    'Gender_Balance': 100,     # F: 남녀 비율
    'Piano_Athlete_Balance': 50,# G: 피아노/운동선호 균등
    'Club_Overlap': 10,        # J: 클럽 활동 편향
    'Score_Deviation': 1       # H: 성적 균등 (평균 총점 편차)
}
# Big M 값 (MIP에서 Indicator Constraint를 선형화할 때 사용)
BIG_M = 200 # 학생 수보다 큰 값

# MIP 솔버 상태를 문자열로 변환하는 헬퍼 함수
def get_solver_status_name(status_code):
    statuses = {
        pywraplp.Solver.OPTIMAL: 'OPTIMAL (최적)',
        pywraplp.Solver.FEASIBLE: 'FEASIBLE (만족스러운 해)',
        pywraplp.Solver.INFEASIBLE: 'INFEASIBLE (해가 없음)',
        pywraplp.Solver.UNBOUNDED: 'UNBOUNDED (무한 해)',
        pywraplp.Solver.ABNORMAL: 'ABNORMAL (비정상 종료)',
        pywraplp.Solver.NOT_SOLVED: 'NOT_SOLVED (해결되지 않음)',
        pywraplp.Solver.MODEL_INVALID: 'MODEL_INVALID (모델 오류)'
    }
    return statuses.get(status_code, 'UNKNOWN')

def solve_class_assignment(df):
    """
    Google OR-Tools Linear Solver (CBC)를 사용하여 MIP 모델을 해결합니다.
    """
    # CBC Solver 초기화
    solver = pywraplp.Solver.CreateSolver('CBC')
    if not solver:
        print("오류: CBC 솔버를 초기화할 수 없습니다. OR-Tools가 MIP/CBC를 지원하는지 확인하세요.")
        return

    # --- 데이터 열에 대한 결측치 처리 (문자열 연산 오류 방지) ---
    # 문자열 처리가 필요한 열의 NaN을 빈 문자열로 대체
    cols_to_fill = ['Leadership', 'Piano', '비등교', '운동선호']
    for col in cols_to_fill:
        df[col] = df[col].fillna('')
    # 'sex' 컬럼도 혹시 모를 상황에 대비해 fillna('unknown') 처리 (현재 데이터에는 NaN 없음)
    df['sex'] = df['sex'].fillna('unknown')


    # 데이터 전처리 및 인덱스 맵핑
    student_ids = df['id'].tolist()
    student_to_index = {sid: i for i, sid in enumerate(student_ids)}
    
    # --- 2. 결정 변수 정의 (Decision Variables) ---
    
    # assignment[s, c]: 학생 s가 학급 c에 배정되면 1, 아니면 0인 이진 변수
    assignment = {}
    for s in range(NUM_STUDENTS):
        for c in CLASS_NAMES:
            # s는 학생 인덱스 (0~199), c는 학급 번호 (1~6)
            assignment[s, c] = solver.BoolVar(f'assign_s{student_ids[s]}_c{c}')

    # --- 3. 하드 제약 조건 (Hard Constraints - 절대 불변) ---
    
    # A. 모든 학생은 단 하나의 학급에 배정되어야 합니다. (필수)
    for s in range(NUM_STUDENTS):
        solver.Add(sum(assignment[s, c] for c in CLASS_NAMES) == 1)

    # --- 4. Soft Constraint 페널티 변수 및 제약 설정 ---
    
    objective_terms = []
    
    # A. Constraint C: 나쁜 관계 배제 (페널티 20000)
    for s_index in range(NUM_STUDENTS):
        row = df.iloc[s_index]
        enemy_id_str = row['나쁜관계']
        
        if enemy_id_str and int(enemy_id_str) in student_to_index:
            e_index = student_to_index[int(enemy_id_str)]
            
            for c in CLASS_NAMES:
                # violation_C[s, e, c]: s와 e가 c반에 함께 배정되면 1
                violation_C = solver.BoolVar(f'viol_C_{s_index}_{e_index}_c{c}') 
                
                # violation_C == 1 <=> assignment[s, c] + assignment[e_index, c] == 2
                # Linearization of AND: violation_C >= a + b - 1 and violation_C <= a and violation_C <= b
                solver.Add(violation_C >= assignment[s_index, c] + assignment[e_index, c] - 1)
                solver.Add(violation_C <= assignment[s_index, c])
                solver.Add(violation_C <= assignment[e_index, c])
                
                objective_terms.append(violation_C * WEIGHTS['Bad_Relation'])


    # B. Constraint D: 멘토-비등교 쌍 같은 반 배정 (페널티 10000)
    for s_index in range(NUM_STUDENTS):
        row = df.iloc[s_index]
        
        # '비등교'는 위에서 이미 ''으로 채워져 있으므로 안전함
        if row['비등교'].lower() == 'yes' and row['좋은관계'] and int(row['좋은관계']) in student_to_index:
            mentor_id = int(row['좋은관계'])
            m_index = student_to_index[mentor_id]
            
            # separation_score: 멘토와 학생이 얼마나 다른 반에 배정되었는지의 총합 (|a-b|의 합)
            separation_score = solver.NumVar(0, 2, f'sep_score_{s_index}_{m_index}')
            
            abs_diff_vars = []
            for c in CLASS_NAMES:
                diff_var = solver.NumVar(-1, 1, f'diff_sm_{s_index}_{m_index}_c{c}')
                solver.Add(diff_var == assignment[s_index, c] - assignment[m_index, c])
                
                # Absolute value linear approximation: abs_diff >= diff, abs_diff >= -diff
                abs_diff = solver.NumVar(0, 1, f'abs_diff_sm_{s_index}_{m_index}_c{c}')
                solver.Add(abs_diff >= diff_var)
                solver.Add(abs_diff >= -diff_var)
                abs_diff_vars.append(abs_diff)
            
            # separation_score == sum_c abs_diff
            solver.Add(separation_score == sum(abs_diff_vars))
            
            objective_terms.append(separation_score * WEIGHTS['Mentor_Separation'])
                
    # C. Constraint E: 리더십 최소 1명 미달 페널티 (페널티 15000)
    leader_indices = df[df['Leadership'].str.lower() == 'yes'].index.tolist()
    
    for c in CLASS_NAMES:
        # leader_count: c반의 리더 수
        leader_count = solver.IntVar(0, CLASS_CAPACITIES[c], f'leader_count_c{c}') 
        solver.Add(leader_count == sum(assignment[s, c] for s in leader_indices))
        
        # is_zero: leader_count == 0 이면 1이 되는 이진 변수 (위반 여부)
        is_zero = solver.BoolVar(f'is_zero_leader_c{c}')
        
        # Indicator Constraint (MIP friendly Big M approach)
        # 1. leader_count >= 1 => is_zero = 0
        solver.Add(leader_count >= 1 - BIG_M * is_zero)
        
        # 2. leader_count = 0 => is_zero = 1 
        # 명시적인 제약: leader_count <= BIG_M * (1 - is_zero)
        solver.Add(leader_count <= BIG_M * (1 - is_zero))
        
        objective_terms.append(is_zero * WEIGHTS['Leader_Violation'])
        

    # D. Constraint B: 정원 33/34명 편차 최소화 (페널티 500)
    for c in CLASS_NAMES:
        target_cap = CLASS_CAPACITIES[c]
        
        actual_cap = solver.IntVar(0, NUM_STUDENTS, f'actual_cap_c{c}') 
        solver.Add(actual_cap == sum(assignment[s, c] for s in range(NUM_STUDENTS)))

        deviation = solver.NumVar(-NUM_STUDENTS, NUM_STUDENTS, f'cap_dev_c{c}') 
        solver.Add(deviation == actual_cap - target_cap)

        # Absolute value linear approximation: abs_deviation >= deviation, abs_deviation >= -deviation
        abs_deviation = solver.NumVar(0, NUM_STUDENTS, f'cap_abs_dev_c{c}') 
        solver.Add(abs_deviation >= deviation)
        solver.Add(abs_deviation >= -deviation)
        
        objective_terms.append(abs_deviation * WEIGHTS['Capacity_Deviation'])

    # E. Constraint I & J: 전년도 학급 중복 및 클럽 활동 편향 최소화 (Soft)
    for col, weight in [('24년 학급', WEIGHTS['Prev_Class_Overlap']), ('클럽', WEIGHTS['Club_Overlap'])]:
        for category in df[col].unique():
            category_indices = df[df[col] == category].index.tolist()
            if not category_indices: continue

            # count_in_c: c반에 배정된 해당 카테고리 학생 수
            for c in CLASS_NAMES:
                count_in_c = solver.IntVar(0, CLASS_CAPACITIES[c], f'{col}_count_c{c}_cat{category}') 
                solver.Add(count_in_c == sum(assignment[s, c] for s in category_indices))
                
                # 해당 카테고리 수의 합계를 최소화 (편향 최소화)
                objective_terms.append(count_in_c * weight)

    # F. Constraint F, G, H: 성적/남녀/특기사항 균등 배분 (Soft)
    
    # 성적 (H)
    scores = df['score'].values
    total_mean_score = scores.mean()
    
    # 평균 성적의 분산을 최소화하는 것이 목표. 각 반의 총점의 편차를 최소화.
    target_total_score = total_mean_score * (NUM_STUDENTS / NUM_CLASSES)
    
    for c in CLASS_NAMES:
        total_score_in_c = solver.IntVar(0, CLASS_CAPACITIES[c] * 100, f'total_score_c{c}') 
        score_terms = [assignment[s, c] * scores[s] for s in range(NUM_STUDENTS)]
        solver.Add(total_score_in_c == sum(score_terms))
        
        deviation = solver.NumVar(-3500, 3500, f'score_dev_c{c}') 
        solver.Add(deviation == total_score_in_c - target_total_score)

        abs_deviation = solver.NumVar(0, 3500, f'score_abs_dev_c{c}') 
        solver.Add(abs_deviation >= deviation)
        solver.Add(abs_deviation >= -deviation)
        
        objective_terms.append(abs_deviation * WEIGHTS['Score_Deviation'])


    # 남녀 비율 (F), 비등교/피아노/운동선호 (G)
    balance_features = {
        'sex_boy': ('sex', 'boy', WEIGHTS['Gender_Balance']),
        '비등교_yes': ('비등교', 'yes', WEIGHTS['Non_Attendance_Balance']),
        'Piano_yes': ('Piano', 'yes', WEIGHTS['Piano_Athlete_Balance']),
        '운동선호_yes': ('운동선호', 'yes', WEIGHTS['Piano_Athlete_Balance']),
    }
    
    for feature_name, (col, val, weight) in balance_features.items():
        # 'sex' 컬럼도 혹시 모를 상황에 대비해 .lower() 사용 전에 처리
        feature_indices = df[df[col].str.lower() == val.lower()].index.tolist()
        if not feature_indices: continue
        
        target_count = len(feature_indices) / NUM_CLASSES
        
        for c in CLASS_NAMES:
            count_in_c = solver.IntVar(0, CLASS_CAPACITIES[c], f'{feature_name}_count_c{c}') 
            solver.Add(count_in_c == sum(assignment[s, c] for s in feature_indices))
            
            deviation = solver.NumVar(-CLASS_CAPACITIES[c], CLASS_CAPACITIES[c], f'{feature_name}_dev_c{c}') 
            solver.Add(deviation == count_in_c - target_count)

            abs_deviation = solver.NumVar(0, CLASS_CAPACITIES[c], f'{feature_name}_abs_dev_c{c}') 
            solver.Add(abs_deviation >= deviation)
            solver.Add(abs_deviation >= -deviation)
            
            objective_terms.append(abs_deviation * weight)


    # --- 5. 목표 함수 설정 (Objective Function Definition) ---

    # 모든 페널티 항의 합을 최소화
    solver.Minimize(sum(objective_terms))


    # --- 6. 솔버 실행 및 결과 처리 ---
    
    solver.SetTimeLimit(60000) # 60초 제한
    
    print("--- MIP 모델 해결 시작 (CBC Solver) ---")
    status = solver.Solve()
    
    print(f"솔버 상태: {get_solver_status_name(status)}") # 수정된 get_solver_status_name 함수 호출

    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        print(f"최종 목표 함수 값 (총 페널티): {solver.Objective().Value()}")
        
        # 결과 저장
        results = []
        for s in range(NUM_STUDENTS):
            assigned_class = -1
            for c in CLASS_NAMES:
                # MIP 솔버는 실수 값을 반환하므로 0.5보다 큰지 확인
                if assignment[s, c].solution_value() > 0.5: 
                    assigned_class = c
                    break
            results.append({'id': student_ids[s], 'new_class': assigned_class})

        results_df = pd.DataFrame(results)
        final_df = df.merge(results_df, on='id', how='left')
        
        # 필요한 컬럼만 선택하고 저장
        final_df = final_df[['id', 'name', 'sex', 'score', '24년 학급', '클럽', 'new_class', 'Leadership', 'Piano', '비등교', '운동선호', '좋은관계', '나쁜관계']]
        final_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
        
        print(f"\n성공적으로 배정 완료. 결과는 '{OUTPUT_FILE}'에 저장되었습니다.")
        
        # 검증 출력
        print("\n--- 학급별 배정 결과 요약 ---")
        summary = []
        for c in CLASS_NAMES:
            class_data = final_df[final_df['new_class'] == c]
            
            # str.lower().eq('yes') 대신 str.lower() == 'yes' 사용
            summary.append({
                'Class': c,
                'Size': len(class_data),
                'Target': CLASS_CAPACITIES[c],
                'Deviation': len(class_data) - CLASS_CAPACITIES[c],
                'Leadership': class_data['Leadership'].str.lower().eq('yes').sum(),
                'Boys': class_data[class_data['sex'] == 'boy'].shape[0],
                'Girls': class_data[class_data['sex'] == 'girl'].shape[0],
                'Piano': class_data['Piano'].str.lower().eq('yes').sum(),
                'Non-Att': class_data['비등교'].str.lower().eq('yes').sum(),
                'Avg. Score': f'{class_data['score'].mean():.2f}'
            })
            
        summary_df = pd.DataFrame(summary)
        print(summary_df.to_markdown(index=False))

    else:
        print("\n최적의 해 또는 만족스러운 해를 찾지 못했습니다.")
        print(f"솔버 상태: {get_solver_status_name(status)}")
        print("하드 제약 조건이 너무 엄격하여 해가 존재하지 않을 수 있습니다.")


if __name__ == '__main__':
    if not os.path.exists(INPUT_FILE):
        print(f"오류: 입력 파일 '{INPUT_FILE}'을 찾을 수 없습니다. 파일을 같은 디렉토리에 넣어주세요.")
    else:
        data_df = pd.read_csv(INPUT_FILE)
        
        # 관계 ID를 정수형으로 변환하기 전에 NaN을 처리 (MIP 모델에서 사용)
        data_df['좋은관계'] = data_df['좋은관계'].fillna('')
        data_df['나쁜관계'] = data_df['나쁜관계'].fillna('')
        
        solve_class_assignment(data_df)
