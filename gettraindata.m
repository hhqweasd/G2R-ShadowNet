%% generate the training data |生成训练数据
clear;close all;clc

% mask directory|掩膜路径
maskdir = 'C:\资料\2020-09-G2R\ISTD_Dataset\train\train_B\';
MD = dir([maskdir '/*.png']);

% ground truth directory|GT路径
shadowdir = 'C:\资料\2020-09-G2R\ISTD_Dataset\train\train_A/';
SD = dir([shadowdir '/*.png']);   

% train_A means Rn.
% train_B means M.
% train_C means S (original shadow image in ISTD)
% train_D means Rs.
% train_E means (S - Rn).
% train_F means ψ(M).
trainshadowDir = 'C:\Users\Administrator\Desktop\PAISTD8\train_A\';
trainmaskDir = 'C:\Users\Administrator\Desktop\PAISTD8\train_B\';
trainsDir = 'C:\Users\Administrator\Desktop\PAISTD8\train_D\';
trainnonshadowDir = 'C:\Users\Administrator\Desktop\PAISTD8\train_E\';
diDir = 'C:\Users\Administrator\Desktop\PAISTD8\train_F\';
mkdir(trainsDir);
mkdir(trainshadowDir);
mkdir(trainnonshadowDir);
mkdir(trainmaskDir);
mkdir(diDir);

% %% generate non-shadow region
% nsDir = 'C:\Users\Administrator\Desktop\PAISTD8\train_I\';
% mkdir(nsDir);
%%
% ISTD dataset image size 480*640
mask = ones([480,640]);
for i=1:size(SD)
    sname = strcat(shadowdir,SD(i).name); 
    mname = strcat(maskdir,MD(i).name); 
    s=imread(sname);
    m=imread(mname);
    nmask=~m;       
    smask=~nmask;   
    s = double(s)/255;
    
%     %% generate non-shadow region
%     nsname=[nsDir,SD(i).name];
%     
%     nmask =repmat(nmask,[1,1,3]);
%     imwrite(s.*double(nmask),nsname); 
    
    %%
    newmname = strcat(maskdir,MD(unidrnd(size(MD,1))).name); 
    newm=imread(newmname);
    newnmask=~newm;       
    newsmask=~newnmask;   
    newmask=setdiff(uint8(newsmask),uint8(smask));
    k=0;
    while ((sum(newmask(:))/sum(smask(:))))<0.8 || ((sum(newmask(:))/sum(smask(:))))>1.2
        newmname = strcat(maskdir,MD(unidrnd(size(MD,1))).name); 
        newm=imread(newmname);
        newnmask=~newm;       
        newsmask=~newnmask;   
        newmask=double(newsmask)-double(newsmask&smask);
        k=k+1;
        if k>100
            break;
        end
        i
    end
    newmask =repmat(newmask,[1,1,3]);
    smask=repmat(smask,[1,1,3]);
    newmaskdia=imdilate(newmask,ones(50));
    
    trainshadowname=[trainshadowDir,SD(i).name];
    trainmaskname=[trainmaskDir,SD(i).name];
    trainsname=[trainsDir,SD(i).name];
    trainnonshadowname=[trainnonshadowDir,SD(i).name];
    diname=[diDir,SD(i).name];
    
    imwrite(s.*double(newmask),trainshadowname); 
    imwrite(double(newmask),trainmaskname); 
    imwrite(s.*double(smask),trainsname);
    imwrite(s.*double(~newmask),trainnonshadowname); 
    imwrite(double(newmaskdia),diname); 
    

end