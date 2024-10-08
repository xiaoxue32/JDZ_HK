input_img = imread("D:\paper Ⅴ\Wafer-data\黑胶灰度改.jpg");
% input_img = imread("D:\小茂桌面\381_1.jpg");
if size(input_img, 3) == 3  % 如果图像为RGB格式
    % 将RGB图像转换为灰度图像
    input_img = rgb2gray(input_img);
end

[filtered_img,img_padded] = curvature_based_truncated_median_filter(input_img, 1,2,0.75);
y =uint8(filtered_img);
figure;
subplot(121);imshow(y),title('均值滤波后图像');
subplot(122);imshow(input_img),title('原图');
[BW,maskedRGBImage] = Segm23(y);
MAE=mae(input_img,y);
PSNR=psnr(y,input_img);
%SSIM=ssim(y,input_img);
%IFC=ifcvec(y,input_img);
%RMSE=grayRMSE(input_img,y);
%EME=eme(double(y),190,5);
%ACC=calculatePixelAccuracy(input_img,y);
%sn=SNR2(input_img,y);

%[iou,yl]= Calc_IOU(y, BW);
%核心模块一：变尺度截断均值滤波特征增强
folderName = 'D:\paper Ⅴ\Wafer-data'; % 绝对路径  
fileName = '滤波图像.png';  
fullFilePath = fullfile(folderName, fileName);  
imwrite(y, fullFilePath);  
disp(['图像已保存到：', fullFilePath]);
function [filtered_img,img_padded] = curvature_based_truncated_median_filter(input_img, min_radius, max_radius, threshold)
[rows, cols] = size(input_img);
filtered_img = double(input_img);
% 边界扩展处理
img_padded = padarray(input_img, [max_radius, max_radius], 'symmetric', 'both');
img_padded=double(img_padded);
% 确定中心点
center = floor((min_radius+max_radius)/2);
% 用于曲率估计的二阶导数滤波器
dxx = [1 -2 1];
dyy = dxx';
result=[];
result2=[];
for i = 1:rows
    for j = 1:cols
        % 计算二阶导数
        img_window = double(img_padded(i:i+2*max_radius, j:j+2*max_radius));
        dxx_img_window = conv2(img_window, dxx, 'same');
        dyy_img_window = conv2(img_window, dyy, 'same');
        % 计算局部曲率
        curvature = sqrt(dxx_img_window(center, center)^2 + dyy_img_window(center, center)^2);
        % 确定窗口半径
%         window_radius =min_radius + (max_radius - min_radius) * (1 - curvature/max(max(curvature(:)), epsilon));
  curvature_normalized = 1 / (1 + exp(-curvature));
%  curvature_normalized = tanh(1 * curvature);
 window_radius = round(min_radius + (max_radius - min_radius) * curvature_normalized);
%        window_radius = round(min_radius + (max_radius - min_radius) * (1 - (curvature/max(max(curvature(:)), epsilon))^beta));
        % 提取圆形窗口
        [CC, RR] = meshgrid(1:2*window_radius+1, 1:2*window_radius+1);
        circle_window = sqrt((CC - window_radius - 1).^2 + (RR - window_radius - 1).^2) <= window_radius;
        patch = img_padded(i:i+(2*window_radius), j:j+(2*window_radius)) .* double(circle_window);
        result = [result,window_radius];
        
        %result2=[ result2,a];
%  %% 进行截断均值滤波
      patch = nonzeros(patch);  % 提取非零元素
        mu = mean(patch);
        sigma = std(patch);
        if abs(filtered_img(i, j) - mu) > threshold * sigma
            filtered_img(i, j) = mu;
        end
    end
end
end
%核心模块二：检测驱动活动轮廓模型特征分割
function [BW, maskedRGBImage, numComponents,cc] = Segm23(Ismooth)
% 边缘检测使用Canny检测器
BWedges = edge(Ismooth, 'sobel');
% 使用bwconncomp查找连接的组件
cc = bwconncomp(BWedges); 
% 保持连接的组件具有足够数量的像素(例如，大于50)
numPixelsThreshold = 50;
numPixels = cellfun(@numel, cc.PixelIdxList);
largeCCIdx = find(numPixels >= numPixelsThreshold);
numLargeComponents = length(largeCCIdx); 
% 将大型连接组件组合成单个二值图像
    BW = false(size(BWedges));
for idx = 1:length(largeCCIdx)
    BW(cc.PixelIdxList{largeCCIdx(idx)}) = true;
end
 
%初始化图形以实现可视化
hFig = figure;
% 手动执行迭代
numIterations = 60;
for iter = 1:numIterations
    % 对活动轮廓进行一次迭代
    BW = activecontour(Ismooth, BW, 5, 'Chan-Vese');
    % 每10次迭代显示轮廓和图像
    if mod(iter, 30) == 0 || iter == 1
        % 可视化分割的当前状态
        figure(hFig);
        subplot(1, 2, 1);
        imshow(Ismooth);
        hold on;
        visboundaries(BW, 'Color', 'r'); % 可视化界面
        hold off;
        title(['Iteration ' num2str(iter)]);
        drawnow; % 更新图形窗口
    end
end
%
% 膨胀操作
BW=bwmorph(BW, 'dilate');   
se=strel('arbitrary',1);
BW=imdilate(BW, se);  
% 计算ROI内部像素值的数量（即值为true的像素数量）  
roiPixelCount = sum(BW(:)); % 将BW转换为列向量并求和  
%imshow(BW),title('膨胀BW')
%将灰度图像转换回RGB格式
X = repmat(Ismooth, [1, 1, 3]);
%掩模RGB图像，只保持ROI
maskedRGBImage = X;
maskedRGBImage(repmat(~BW, [1 1 3])) = 0;
fprintf('ROI中的像素数: %d\n', roiPixelCount);  
cc_total = bwconncomp(BW);  
numComponents = cc_total.NumObjects;  
fprintf('阈值后的大组件数: %d\n', numLargeComponents);  
fprintf('最终BW图像中组件的总数: %d\n', numComponents);  
% 可视化
figure;
subplot(1, 2, 1);
imshow(BW);
title('BW Image');
subplot(1, 2, 2);
imshow(maskedRGBImage);
title('Masked RGB Image');
% 
end
