% 读取图像  
try  
    img = imread("D:\paper Ⅴ\Wafer-data\出图\线终图.png"); % 替换为你的图像文件名  
catch ME  
    error('无法读取图像文件: %s', ME.message);  
end  
  
% 如果图像不是灰度图像，转换为灰度图像  
if size(img, 3) == 3  
    img_gray = rgb2gray(img);  
else  
    img_gray = img;  
end  
  
% 设定你想要更改的灰度范围，例如从100到200（注意MATLAB中灰度范围是0-255）  
gray_range_min = 70;  
gray_range_max = 120;  
  
% 创建一个新的RGB图像，与原始图像大小相同，用于存储结果  
% 注意这里需要将size(img_gray)转换为行向量  
img_colored = zeros([size(img_gray, 1), size(img_gray, 2), 3], 'uint8');  
  
% 初始化计数器  
replaced_pixels = 0;  
  
% 遍历图像中的每个像素  
for y = 1:size(img_gray, 1)  
    for x = 1:size(img_gray, 2)  
        % 检查像素的灰度值是否在指定范围内  
        if img_gray(y, x) >= gray_range_min && img_gray(y, x) <= gray_range_max  
            % 生成随机颜色（RGB值在0-255之间）  
            random_color = randi([0, 255], 1, 3, 'uint8');  
            % 将随机颜色赋值给结果图像的对应像素  
            img_colored(y, x, :) = random_color;  
            % 增加计数器  
            replaced_pixels = replaced_pixels + 1;  
        end  
    end  
end  
  
% 显示结果图像  
imshow(img_colored);  
  
% 输出替换的像素数量  
fprintf('已替换的像素数量: %d\n', replaced_pixels);

  
% 展示结果图像（可选）  
imshow(img_colored);  
  
% 创建一个二值图像，其中彩色区域为1（白色），黑色背景为0（黑色）  
% 假设彩色区域对应于img_gray中灰度值在gray_range_min和gray_range_max之间的像素  
bw_img = (img_gray >= gray_range_min) & (img_gray <= gray_range_max);  
  
% 查找二值图像中的连通分量  
[L, numRegions] = bwlabel(bw_img);  
  
% 输出被黑色背景完全包围的彩色区域数量（减去背景区域）  
fprintf('被黑色背景完全包围的彩色区域数量: %d\n', numRegions - 1);  
  
% 可视化连通分量（可选）  
figure;  
imshow(label2rgb(L, 'jet', 'k'));  
title('连通分量可视化');