% DACE toolbox 연결
addpath('C:\Users\Documents\MATLAB\dace')

%% 1. 초기 데이터 불러오기
disp('--- 1. 초기 데이터 불러오기 ---');
data = readtable('C:\Users\파일위치\파일이름.확장자', 'VariableNamingRule', 'preserve');
X_initial = table2array(data(시행횟수, 파라미터개수));
Y_initial = table2array(data(시행횟수:, 목적함수개수));

valid_idx = all(~isnan([X_initial Y_initial]), 2);
X_initial = X_initial(valid_idx, :);
Y_initial = Y_initial(valid_idx, :);

num_vars = size(X_initial, 2);

% 변수 범위 정의
lb = zeros(1, num_vars); ub = zeros(1, num_vars);
lb = lb(x)
ub = ub(x)
% 변수의 가장 큰 값과 작은 값이 boundary 값이 됨

%% 2. 적응형 샘플링 및 최적화 루프
disp('--- 2. 적응형 샘플링 및 최적화 시작 ---');

% --- 루프 설정 ---
max_iterations = 20; % 최대 반복 횟수
num_new_samples_per_iter = 10; % 반복당 추가할 샘플 수
%수가 커질수록 시간이 많이 걸림

current_X = X_initial;
current_Y = Y_initial;

% --- GA 옵션 수정 (더 넓고 촘촘하게 탐색) ---
opts = optimoptions('gamultiobj', ...
    'Display','off', ... 
    'PopulationSize', 350, ... % 모집단 크기
    'MaxGenerations', 300, ... % 세대 수
    'FunctionTolerance', 1e-5, ... % iteration 종료 기준
    'PlotFcn', 'gaplotpareto', ... % pareto 변화 추이 확인
    'DistanceMeasureFcn', {@distancecrowding, 'phenotype'}, ...
    'ParetoFraction', 0.5); % 파레토 해 집단 비율 명시적 증가 (기본값 0.35 -> 0.5)

for iter = 1:max_iterations
    fprintf('\n--- 반복 %d/%d ---\n', iter, max_iterations);

    % --- 2.1. 현재 데이터로 Kriging 모델 학습 ---
    disp('Kriging 모델 학습 중...');
    theta_init = ones(1, num_vars) * 10;
    theta_lob = 1e-3 * ones(1, num_vars);
    theta_upb = 1e3 * ones(1, num_vars);
    
    [dmodel_obj1, ~] = dacefit(current_X, current_Y(:,1), @regpoly0, @corrgauss, theta_init, theta_lob, theta_upb);
    [dmodel_obj2, ~] = dacefit(current_X, current_Y(:,2), @regpoly0, @corrgauss, theta_init, theta_lob, theta_upb);
    [dmodel_obj3, ~] = dacefit(current_X, current_Y(:,3), @regpoly0, @corrgauss, theta_init, theta_lob, theta_upb);

    % --- 2.2. 현재 모델로 최적화 수행 (짧게) ---
    disp('다목적 최적화 수행 중...');
    Y_min = min(current_Y, [], 1);
    Y_max = max(current_Y, [], 1);
    normalize_obj = @(vals) (vals - Y_min) ./ (Y_max - Y_min + eps);
    
    objfun_norm_pred = @(x) normalize_obj([predictor(x, dmodel_obj1), predictor(x, dmodel_obj2), predictor(x, dmodel_obj3)]);
    final_obj_for_ga = @(x) final_obj_helper(x, objfun_norm_pred);
    
    % 초기 모집단을 현재 데이터로 설정하여 안정성 확보
    opts.InitialPopulationMatrix = current_X(randi(size(current_X,1), min(size(current_X,1), 100)), :);
    [x_pareto, ~] = gamultiobj(final_obj_for_ga, num_vars, [], [], [], [], lb, ub, [], opts);

    % --- 2.3. 다음 샘플링 지점 결정 (불확실성 기반) ---
    disp('다음 샘플링 지점 결정 중...');
    new_X_samples = zeros(num_new_samples_per_iter, num_vars);
    
    % 후보군: 현재 파레토 해 + 탐색 공간의 랜덤 포인트
    num_candidates = 1000;
    candidates = [x_pareto; lb + rand(num_candidates, num_vars) .* (ub - lb)];
    
    % 각 후보의 불확실성(분산) 계산
    [~, mse1] = predictor(candidates, dmodel_obj1);
    [~, mse2] = predictor(candidates, dmodel_obj2);
    [~, mse3] = predictor(candidates, dmodel_obj3);
    
    % 불확실성을 정규화하고 합산하여 총 불확실성 점수 계산
    total_uncertainty = normalize(mse1) + normalize(mse2) + normalize(mse3);
    
    % 기존 샘플과 너무 가까운 후보는 제외 (중복 방지)
    min_dist_to_existing = min(pdist2(candidates, current_X), [], 2);
    total_uncertainty(min_dist_to_existing < 1e-3) = -inf; % 매우 낮은 점수 부여
    
    % 가장 불확실성이 높은 지점들을 새로운 샘플로 선택
    [~, sorted_idx] = sort(total_uncertainty, 'descend');
    new_X_samples = candidates(sorted_idx(1:num_new_samples_per_iter), :);

    % --- 2.4. 새로운 샘플 평가 및 데이터셋 업데이트 ---
    % !!! 중요: 이 부분은 실제 시뮬레이션/실험을 호출하는 로직으로 대체되어야 합니다.
    % 여기서는 임시로 Y_initial에서 무작위로 값을 가져와 모사합니다.
    disp('새로운 샘플 평가 및 데이터셋 업데이트...');
    % new_Y_samples = 실제_함수_호출(new_X_samples);
    new_Y_samples = Y_initial(randi(size(Y_initial, 1), num_new_samples_per_iter, 1), :); 

    current_X = [current_X; new_X_samples];
    current_Y = [current_Y; new_Y_samples];
    
    fprintf('현재 총 샘플 수: %d\n', size(current_X, 1));
end

disp('--- 적응형 샘플링 및 최적화 완료 ---');

%% 3. 최종 결과 처리 및 시각화
disp('--- 3. 최종 결과 처리 ---');

% --- Pareto Front 식별 로직 수정 ---
% paretomember 함수를 대체하는 직접 구현 로직
n_points = size(current_Y, 1);
is_pareto = true(n_points, 1); % 모든 점이 파레토 해라고 가정하고 시작

for i = 1:n_points
    % i번째 점이 다른 점 j에 의해 지배되는지 확인
    for j = 1:n_points
        if i == j, continue; end
        
        % 지배 조건 확인 (obj1: 최대화, obj2/3: 최소화)
        % j가 i를 지배하려면, 모든 목적함수에서 나쁘지 않고, 하나 이상에서 명백히 좋아야 함
        if (current_Y(j,1) >= current_Y(i,1) && current_Y(j,2) <= current_Y(i,2) && current_Y(j,3) <= current_Y(i,3)) && ...
           (current_Y(j,1) >  current_Y(i,1) || current_Y(j,2) <  current_Y(i,2) || current_Y(j,3) <  current_Y(i,3))
            is_pareto(i) = false; % i는 지배당했으므로 파레토 해가 아님
            break; % 더 이상 확인할 필요 없음
        end
    end
end

x_pareto_final = current_X(is_pareto, :);
fval_pareto_final = current_Y(is_pareto, :);

% Pareto 결과를 표로 정리
num_cases = size(x_pareto_final, 1);
case_names = strcat('case', string(1:num_cases))';
Para_names = strcat('Para', string(1:num_vars));
obj_names_orig = {'Effectiveness', 'MeanStress', 'MaxStress'};
table_names = ['Case', Para_names, obj_names_orig];

pareto_table = array2table([case_names, num2cell(x_pareto_final), num2cell(fval_pareto_final)], ...
    'VariableNames', table_names);

disp('최종 Pareto 결과 표 (원래 스케일):');
disp(pareto_table);

 try
    filename = '파일이름.확장자(예:.xlsx)';
    writetable(pareto_table, filename, 'Sheet', 'Pareto_Solutions', 'WriteMode', 'overwrite');
    fprintf('\n "%s" 파일로 내보내기가 완료되었습니다.\n', filename);
    fprintf('MATLAB 현재 폴더에서 파일을 확인하세요.\n\n');
 catch ME
    fprintf('\n 엑셀 파일 저장 중 오류가 발생했습니다.\n');
    fprintf('오류 메시지: %s\n', ME.message);
 end

% Pareto front 시각화
figure;
scatter3(fval_pareto_final(:,1), fval_pareto_final(:,2), fval_pareto_final(:,3), 60, 'filled');
xlabel('목표함수1');
ylabel('목표함수2');
zlabel('목표함수3');
title('Final Pareto Front');
grid on;
view(-30, 20);

%% 헬퍼 함수
function out = final_obj_helper(x, objfun)
    norm_vals = objfun(x);
    out = [-norm_vals(1), norm_vals(2), norm_vals(3)];
end
