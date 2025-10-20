%% 1. Створення штучних даних (383 спостереження)
rng(1); % фіксуємо генератор для відтворюваності
n = 383;

chol     = 207 + 44*randn(n,1);
stab_glu = 107 + 53*randn(n,1);
hdl      = 50 + 17*randn(n,1);
ratio    = 4.52 + 1.75*randn(n,1);
glyhb    = 5.57 + 2.2*randn(n,1);
height   = 66 + 3.9*randn(n,1);
weight   = 177 + 40*randn(n,1);

data = table(chol, stab_glu, hdl, ratio, glyhb, height, weight);

%% 2. Побудова лінійної регресійної моделі
mdl = fitlm(data, 'chol ~ stab_glu + hdl + ratio + glyhb + height + weight');

%% 3. Вивід результатів
disp(mdl)

%% 4. Збереження передбачених значень та залишків
data.Predicted = mdl.Fitted;                  % передбачені значення
data.Residuals = mdl.Residuals.Raw;           % нестандартизовані залишки
data.StdResiduals = mdl.Residuals.Standardized; % стандартизовані залишки
data.StudentizedResiduals = mdl.Residuals.Studentized; % стьюдентизовані залишки

%% 5. Графіки залишків
figure;
subplot(2,2,1)
plot(mdl.Fitted, mdl.Residuals.Raw, 'o')
xlabel('Передбачене значення'); ylabel('Залишки')
title('Залишки vs Передбачене'); grid on

subplot(2,2,2)
qqplot(mdl.Residuals.Raw)
title('Нормальний графік залишків');

subplot(2,2,3)
histogram(mdl.Residuals.Standardized, 20)
xlabel('Стандартизовані залишки'); ylabel('Частота')
title('Гістограма стандартизованих залишків');

subplot(2,2,4)
plotDiagnostics(mdl,'leverage'); % графік діагностики
title('Впливові спостереження');

%% 6. Перевірка мультиколінеарності (VIF)
X = [data.stab_glu, data.hdl, data.ratio, data.glyhb, data.height, data.weight];
VIF = zeros(1, size(X,2));
for i = 1:size(X,2)
    Xi = X(:,i);
    Xother = X(:,[1:i-1, i+1:end]);
    R2 = fitlm(Xother, Xi).Rsquared.Ordinary;
    VIF(i) = 1/(1-R2);
end
disp('Variance Inflation Factor (VIF) для змінних:');
disp(array2table(VIF, 'VariableNames', {'stab_glu','hdl','ratio','glyhb','height','weight'}));

%% 7. Збереження результатів у CSV (опційно)
writetable(data,'regression_results.csv');