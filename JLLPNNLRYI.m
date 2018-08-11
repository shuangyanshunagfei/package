function [F,A,Z,E] = JLLPNNLRYI(X,lambda,beta,rho,maxIter)
%% ������������
[dd, nn] = size(X);
tol1 = 1e-6; % threshold for the error in constraint: X-F'*X*Z-E
tol2 = 1e-6; % threshold for the change in constraint: Z-A
max_mu = 1e10;
mu = 1e-3;  % for feature extraction 1e-3 is the best, for subspace segmentation 1e-1 is the best.
options = [];
options.NeighborMode = 'KNN';
options.k = 5;
options.WeightMode = 'HeatKernel'; %HeatKernel
options.t = 1;
options.ReducedDim = dd;
A0 = constructW(X',options);
A0 = (A0'+A0)/2;
L = diag(sum(A0))-A0;

%% �����ʼ��
Z = zeros(nn,nn);
F = eye(dd);
% F = rand(dd,dd);
% F = zeros(dd,dd);
E = sparse(dd,nn);
% E = zeros(dd,nn);
A = zeros(nn,nn);
Y1 = zeros(dd,nn);
Y2 = zeros(nn,nn);

%% ��������
iter=0;
while iter<maxIter
    iter = iter+1;
    
    % ���¾����ֵ
     Z = Update_ZYI(X,Y1,Y2,mu,E,A,F);
% Z=eye(size(X,2),size(X,2));
     A = Update_AYI(Z,Y2,mu);
% A=Z;
%     Z = Update_Z(X,Z,Y1,Y2,mu,E,A,F);
%     A = Update_W(F,X,Z,Y2,mu,lambda);
    F = Update_FYI(L,Z,E,X,Y1,mu,beta);
    E = Update_EYI(X,F,Z,Y1,mu,lambda);
    
%     fprintf('iter=%d: rank(Z)=%d, rank(A)=%d, mu=%.6f\n', iter, rank(Z,1e-4*norm(Z,2)), rank(A), mu);
%     
%     figure(1);
%     subplot(2,2,1), imshow(Z, []), xlabel('Z');
%     subplot(2,2,2), imshow(A, []), xlabel('A');
%     subplot(2,2,3), imshow(F, []), xlabel('F');
%     subplot(2,2,4), imshow(E, []), xlabel('E');
%     drawnow();
%     
%     term1(iter,1) = rank(Z,1e-4*norm(Z,2));
%     term2(iter,1) = beta*trace(F'*X*L*X'*F);
%     term3(iter,1) = 0.0;
%     for i = 1:size(E,2)
%         term3(iter,1) = term3(iter,1) + norm(E(:,i));
%     end
%     term3(iter,1) = lambda*term3(iter,1);
%     obj(iter,1) = term1(iter,1)+term2(iter,1)+term3(iter,1);
%     figure(2);
% %     subplot(2,2,1), plot(1:length(term1),term1), xlabel('rank Z');
% %     subplot(2,2,2), plot(1:length(term2),term2), xlabel('LPP');
% %     subplot(2,2,3), plot(1:length(term3),term3), xlabel('Error');
% %     subplot(2,2,4), plot(1:length(obj),obj), xlabel('Objective');
% 
% hold on;
%     plot(1:length(obj),obj), xlabel('Objective');
%     hold off;
%     xlabel('Iteration Number','fontsize',10);
%     ylabel('Objective Value','fontsize',10);
%     legend('PIE');
%     drawnow();
%     
    % �����������ճ���
    Y1 = Y1+mu*(X-F'*X*Z-E);
     Y2 = Y2+mu*(Z-A);
    
    % ���������ж�
    eq1 = max(max(abs(X-F'*X*Z-E)));
     eq2 = max(max(abs(Z-A)));
     fprintf('\teq1(X-F''*X*Z-E)=%.3e, eq2(Z-A)=%.3e\n', eq1, eq2);
     if eq1<=tol1 && eq2<=tol2
        fprintf('reach the convergence constraint...\n');
        break;
    end
    % ����ѧϰ��
    mu = min(max_mu,mu*rho);
end
