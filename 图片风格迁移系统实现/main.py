from __future__ import print_function
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image, ImageTk
import torchvision.transforms as transforms
import torchvision.models as models
import copy



class Application(object):
    def __init__(self):
        self.root = tk.Tk()
        self.root.geometry('800x500')
        self.root.title('LRZ图片风格迁移系统')



        global photo1
        photo1 = tk.PhotoImage(file=r'bg1.png')
        self.label1= tk.Label(self.root, image=photo1,bg='white')
        self.label1.place(x=40,y=30)
        self.bt1=tk.Button(self.root, text='打开', command=self.getfile1)
        self.bt1.place(x=75,y=140)

        self.symbol1=tk.Label(self.root,text='+',font=("黑体", 40))
        self.symbol1.place(x=175,y=55)

        global photo2
        photo2 = tk.PhotoImage(file=r'bg2.png')
        self.label2 = tk.Label(self.root, image=photo2, bg='white')
        self.label2.place(x=240, y=30)
        self.bt2 = tk.Button(self.root, text='打开', command=self.getfile2)
        self.bt2.place(x=275, y=140)

        self.symbol2 = tk.Label(self.root, text='=', font=("黑体", 40))
        self.symbol2.place(x=375, y=55)

        global photo3
        photo3 = tk.PhotoImage(file=r'bg3.png')
        self.label3 = tk.Label(self.root, image=photo3, bg='white')
        self.label3.place(x=440, y=30)
        self.value = 256
        def get_value1():
            self.value=128
        def get_value2():
            self.value=256

        self.rb1=tk.Radiobutton(self.root,text='128px',variable=1,value=True,command=get_value1)
        self.rb1.place(x=440,y=130)
        self.rb2 = tk.Radiobutton(self.root, text='256px', variable=1, value=False,command=get_value2)
        self.rb2.place(x=500, y=130)
        self.bt3= tk.Button(self.root, text='生成', command=self.to_begin)
        self.bt3.place(x=475, y=160)

        self.bt4 = tk.Button(self.root, text='保存结果', command=self.save_img)
        self.bt4.place(x=620, y=68)

        self.text=tk.Text(self.root,width=100,height=15)
        self.text.place(x=40,y=250)



        self.root.mainloop()


    def getfile1(self):
        file_path = filedialog.askopenfilename(title='选择文件', filetypes=[(('JPG', '*.jpg')), ('All Files', '*')])
        img = Image.open(file_path)
        self.input_img=img.resize((350,350))
        img = img.resize((100, 100))

        global photo1
        photo1 = ImageTk.PhotoImage(img)
        self.label1.configure(image=photo1)
        self.label1.image = photo1

    def getfile2(self):
        file_path = filedialog.askopenfilename(title='选择文件', filetypes=[(('JPG', '*.jpg')), ('All Files', '*')])
        img = Image.open(file_path)
        self.style_img=img.resize((350,350))
        img = img.resize((100, 100))

        global photo2
        photo2 = ImageTk.PhotoImage(img)
        self.label2.configure(image=photo2)
        self.label2.image = photo2


    def to_begin(self):


        migrate_main=Migrate(self.input_img,self.style_img,self.text,self.value)
        out_img=migrate_main.begin()
        self.out_img = out_img
        img = out_img.resize((100, 100))

        global photo3
        photo3 = ImageTk.PhotoImage(img)
        self.label3.configure(image=photo3)
        self.label3.image = photo3

    def save_img(self):
        fname = tk.filedialog.asksaveasfilename(title=u'保存文件', filetypes=[("PNG", ".png")])
        self.out_img.save(str(fname) + '.png', 'PNG')




class Migrate():
    def __init__(self,input_img,style,t,value):
        self.input_img=input_img
        self.style=style
        self.t=t
        self.value=value


    def begin(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        imsize = 512 if torch.cuda.is_available() else self.value  # use small size if no gpu

        loader = transforms.Compose([
            transforms.Resize(imsize),
            transforms.ToTensor()])

        def image_loader(image):

            image = loader(image).unsqueeze(0)
            return image.to(device, torch.float)

        style_img = image_loader(self.style)
        content_img = image_loader(self.input_img)

        assert style_img.size() == content_img.size(), "we need to import style and content images of the same size"


        class ContentLoss(nn.Module):
            def __init__(self, target, ):
                super(ContentLoss, self).__init__()
                self.target = target.clone().detach()

            def forward(self, input):
                self.loss = F.mse_loss(input, self.target)
                return input

        def gram_matrix(input):
            a, b, c, d = input.size()
            features = input.view(a * b, c * d)
            G = torch.mm(features, features.t())
            return G.div(a * b * c * d)

        class StyleLoss(nn.Module):
            def __init__(self, target_feature):
                super(StyleLoss, self).__init__()
                self.target = gram_matrix(target_feature).clone().detach()
            def forward(self, input):
                G = gram_matrix(input)
                self.loss = F.mse_loss(G, self.target)
                return input

        cnn = models.vgg19(pretrained=True).features.to(device).eval()
        cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)


        class Normalization(nn.Module):
            def __init__(self, mean, std):
                super(Normalization, self).__init__()
                self.mean = torch.tensor(mean).view(-1, 1, 1)  #（3，1，1）
                self.std = torch.tensor(std).view(-1, 1, 1)
            def forward(self, img):
                return (img - self.mean) / self.std


        import warnings
        warnings.filterwarnings("ignore")
        normalization = Normalization(cnn_normalization_mean, cnn_normalization_std).to(device)
        model=nn.Sequential(normalization)
        a=model(style_img).clone().detach()
        c=gram_matrix(a)



        content_layers_default = ['conv_4'] #在第四个卷积层后进行计算 内容损失
        style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5'] #在每个卷积层后计算风格损失
        def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                                       style_img, content_img,
                                       content_layers=content_layers_default,
                                       style_layers=style_layers_default):
            cnn = copy.deepcopy(cnn)
            normalization = Normalization(normalization_mean, normalization_std).to(device)
            content_losses = []
            style_losses = []
            model = nn.Sequential(normalization) #先在model 中加上标准化层
            i = 0
            for layer in cnn.children():
                if isinstance(layer, nn.Conv2d):
                    i += 1
                    name = 'conv_{}'.format(i)
                elif isinstance(layer, nn.ReLU):
                    name = 'relu_{}'.format(i)
                    layer = nn.ReLU(inplace=False)
                elif isinstance(layer, nn.MaxPool2d):
                    name = 'pool_{}'.format(i)
                elif isinstance(layer, nn.BatchNorm2d):
                    name = 'bn_{}'.format(i)
                else:
                    raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
                model.add_module(name, layer) #每遇到一个层 就加到model中
                if name in content_layers:
                    target = model(content_img).clone().detach()
                    content_loss = ContentLoss(target)
                    model.add_module("content_loss_{}".format(i), content_loss)
                    content_losses.append(content_loss)
                if name in style_layers:
                    target_feature = model(style_img).clone().detach()
                    style_loss = StyleLoss(target_feature)
                    model.add_module("style_loss_{}".format(i), style_loss)
                    style_losses.append(style_loss)
            for i in range(len(model) - 1, -1, -1):
                if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                    break
            model = model[:(i + 1)]
            return model, style_losses, content_losses

        def write(self, info):
            self.t.insert('end', info)
            self.t.update()
            self.t.see(tk.END)

        input_img = content_img.clone()
        def get_input_optimizer(input_img):
            optimizer = optim.LBFGS([input_img.requires_grad_()])
            return optimizer

        def run_style_transfer(cnn, normalization_mean, normalization_std,
                               content_img, style_img, input_img, num_steps=500,
                               style_weight=1000000, content_weight=1):

            model, style_losses, content_losses = get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                                                                             style_img, content_img)

            optimizer = get_input_optimizer(input_img)

            write(self,'图片迁移学习开始.....  请耐心等待\n')
            print('图片迁移学习开始..')
            run = [0]
            while run[0] <= num_steps:
                def closure():
                    input_img.data.clamp_(0, 1)
                    optimizer.zero_grad()
                    model(input_img)
                    style_score = 0
                    content_score = 0
                    for sl in style_losses:
                        style_score += sl.loss
                    for cl in content_losses:
                        content_score += cl.loss
                    style_score *= style_weight
                    content_score *= content_weight
                    loss = style_score + content_score
                    loss.backward()
                    run[0] += 1
                    if run[0] % 50 == 0:
                        write(self,"run {}/500:\n".format(run))
                        write(self,'Style Loss : {:4f} Content Loss: {:4f}\n'.format(style_score.item(), content_score.item()))
                        write(self," ")
                        print("run {}:".format(run))
                        print('Style Loss : {:4f} Content Loss: {:4f}'.format(style_score.item(), content_score.item()))

                    return style_score + content_score
                optimizer.step(closure)
            input_img.data.clamp_(0, 1)
            return input_img

        output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std, content_img, style_img, input_img)

        unloader = transforms.ToPILImage()
        def imshow(tensor):
            image = tensor.cpu().clone().detach()
            image = image.squeeze(0)
            image = unloader(image)
            plt.imshow(image)
            return image

        write(self,'图片风格迁移成功!!')
        image_out=imshow(output)
        return image_out

app = Application()

