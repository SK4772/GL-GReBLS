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

train_x = zscore(train_x')';%��׼��
H1 = [train_x .1 * ones(size(train_x,1),1)];          %���һ����Ϊƫ����
y=zeros(size(train_x,1),N2*N1);                       %����ӳ��ڵ����
for i=1:N2
    we=2*rand(size(train_x,2)+1,N1)-1;%����ת��Ϊ[-1,1]
    We{i}=we;
    A1 = H1 * we;%����ӳ��ڵ�
    A1 = mapminmax(A1);%��׼��
    clear we;
beta1  =  sparse_bls(A1,H1,1e-3,50)';                  %ϡ���Զ�������΢��Ȩ�ؾ���

beta11{i}=beta1;
% clear A1;
T1 = H1 * beta1;%���յ������ڵ����
fprintf(1,'Feature nodes in window %f: Max Val of Output %f Min Val %f\n',i,max(T1(:)),min(T1(:)));

[T1,ps1]  =  mapminmax(T1',0,1);%ÿ��ӳ�䵽[0,1]����
T1 = T1';
ps(i)=ps1;
% clear H1;
% y=[y T1];
y(:,N1*(i-1)+1:N1*i)=T1;%������ӳ��ڵ�������y
end

clear H1;
clear T1;
%%%%%%%%%%%%%enhancement nodes%%%%%%%%%%%%%%%%%%%%%%%%%%%%

H2 = [y .1 * ones(size(y,1),1)];%����һ��ƫ��
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if N1*N2>=N3
     wh=orth(2*rand(N2*N1+1,N3)-1);%��׼������
else
    wh=orth(2*rand(N2*N1+1,N3)'-1)'; %Ϊʲô�ù���ת��
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
T2 = H2 *wh;%�õ���ǿ�ڵ㣬�õ�һ�ַ���
l2 = max(max(T2));%�õ����ֵ
l2 = s/l2;%Լ��ϵ��
%fprintf(1,'Enhancement nodes: Max Val of Output %f Min Val %f\n',l2,min(T2(:)));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
T2 = tansig(T2 * l2);%Ϊʲô����������ֵ
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
T3=[y T2];%�����������
clear H2;clear T2;
disp('Training has been finished!');



%%%%%%%%%%%%%%%%%%%%%%Testing Process%%%%%%%%%%%%%%%%%%%
test_x = zscore(test_x')';%�Բ������ݱ�׼������
HH1 = [test_x .1 * ones(size(test_x,1),1)];%����ƫ��
%clear test_x;
yy1=zeros(size(test_x,1),N2*N1);%��Ϊ����ӳ��ڵ����
for i=1:N2
    beta1=beta11{i};ps1=ps(i);
    TT1 = HH1 * beta1;
    TT1  =  mapminmax('apply',TT1',ps1)';%��һ������

clear beta1; clear ps1;
%yy1=[yy1 TT1];
yy1(:,N1*(i-1)+1:N1*i)=TT1;%�������ڵ�ֵ����
end
clear TT1;
clear HH1;
HH2 = [yy1 .1 * ones(size(yy1,1),1)]; %����ƫ��
TT2 = tansig(HH2 * wh * l2); %�����
TT3=[yy1 TT2]; %�������
clear HH2;clear wh;clear TT2;

