import openpyxl
from PIL import Image

def main():
    # 读取图像文件
    image = Image.open('232-101.png').convert('L')

    # 创建新的Excel工作簿
    workbook = openpyxl.Workbook()
    sheet = workbook.active

    # 将每个像素的灰度值存储在Excel中
    for y in range(image.size[1]):
        for x in range(image.size[0]):
            pixel = image.getpixel((x, y))
            sheet.cell(row=y + 1, column=x + 1, value=pixel)

    # 保存Excel文件
    workbook.save('232.xlsx')
    print("处理完毕")

if __name__ == '__main__':
    main()