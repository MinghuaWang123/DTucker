%此程序用来计算根据相关性排好序的光谱角
%[msad,SAD] = mSAD(A_true,A_es)
%Input: 
%A_true     为真实的，排好序的端元光谱矩阵
%A_es       为估计的，与真实端元光谱矩阵一一对应的估计端元矩阵
%Output：
%masd       为光谱角平均误差
%SAD        为每条曲线的误差角

function [msad,SAD] = mSAD(A_true,A_es)


[~,p] = size(A_true);

SAD = zeros(1,p);

for i = 1:p
    x = A_true(:,i);
    y = A_es(:,i);
    SAD(i) = acos( x'*y/(norm(x)*norm(y)) )*180/pi;
end

% msad = norm(SAD,'fro')/p;
msad = mean(SAD);

end
