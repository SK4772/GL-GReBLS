function [T3,TT3] = bls_train(train_x,test_x,s,N1,N2,N3)
% Learning Process of the proposed broad learning system
%Input: 
%---train_x,test_x : the training data and learning data 
%---train_y,test_y : the label 
%---We: the randomly generated coefficients of feature nodes
%---wh:the randomly generated coefficients of enhancement nodes
%----s: the shrinkage parameter for enhancement nodes
%----C: the regularization parameter for sparse regualarization
%----N11: the number of feature nodes  per window
%----N2: the number of windows of feature nodes

%%%%%%%%%%%%%%feature nodes%%%%%%%%%%%%%%

train_x = zscore(train_x')';%标准化
H1 = [train_x .1 * ones(size(train_x,1),1)];          %多加一列作为偏移量
y=zeros(size(train_x,1),N2*N1);                       %特征映射节点矩阵
for i=1:N2
    we=2*rand(size(train_x,2)+1,N1)-1;%区间转化为[-1,1]
    We{i}=we;
    A1 = H1 * we;%特征映射节点
    A1 = mapminmax(A1);%标准化
    clear we;
beta1  =  sparse_bls(A1,H1,1e-3,50)';                  %稀疏自动编码器微调权重矩阵

beta11{i}=beta1;
% clear A1;
T1 = H1 * beta1;%最终的特征节点矩阵
fprintf(1,'Feature nodes in window %f: Max Val of Output %f Min Val %f\n',i,max(T1(:)),min(T1(:)));

[T1,ps1]  =  mapminmax(T1',0,1);%每行映射到[0,1]区间
T1 = T1';
ps(i)=ps1;
% clear H1;
% y=[y T1];
y(:,N1*(i-1)+1:N1*i)=T1;%将特征映射节点存入矩阵y
end

clear H1;
clear T1;
%%%%%%%%%%%%%enhancement nodes%%%%%%%%%%%%%%%%%%%%%%%%%%%%

H2 = [y .1 * ones(size(y,1),1)];%加上一列偏差
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if N1*N2>=N3
     wh=orth(2*rand(N2*N1+1,N3)-1);%标准正交基
else
    wh=orth(2*rand(N2*N1+1,N3)'-1)'; %为什么用共轭转置
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
T2 = H2 *wh;%得到增强节点，用第一种方法
l2 = max(max(T2));%得到最大值
l2 = s/l2;%约束系数
%fprintf(1,'Enhancement nodes: Max Val of Output %f Min Val %f\n',l2,min(T2(:)));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
T2 = tansig(T2 * l2);%为什么乘以这个最大值
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
T3=[y T2];%最后的输入矩阵
clear H2;clear T2;
disp('Training has been finished!');



%%%%%%%%%%%%%%%%%%%%%%Testing Process%%%%%%%%%%%%%%%%%%%
test_x = zscore(test_x')';%对测试数据标准化处理
HH1 = [test_x .1 * ones(size(test_x,1),1)];%包含偏差
%clear test_x;
yy1=zeros(size(test_x,1),N2*N1);%作为特征映射节点矩阵
for i=1:N2
    beta1=beta11{i};ps1=ps(i);
    TT1 = HH1 * beta1;
    TT1  =  mapminmax('apply',TT1',ps1)';%归一化处理

clear beta1; clear ps1;
%yy1=[yy1 TT1];
yy1(:,N1*(i-1)+1:N1*i)=TT1;%将特征节点值存入
end
clear TT1;
clear HH1;
HH2 = [yy1 .1 * ones(size(yy1,1),1)]; %包含偏差
TT2 = tansig(HH2 * wh * l2); %激活函数
TT3=[yy1 TT2]; %输入矩阵
clear HH2;clear wh;clear TT2;

