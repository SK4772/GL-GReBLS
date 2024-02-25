function [Y] = vectorTomatirc(labels)
%VECTORTOMATIRC 标签向量转标签矩阵
%   此处显示详细说明
label = unique(labels);
Y = bsxfun(@eq, labels, label');
Y = double(Y);
end

