num=[10,20,30,40];
%parpool(8);
acc=zeros(10,1);
for j=1:10
    acc(j)=GLGREBLS(4);
end
a=mean(acc);
