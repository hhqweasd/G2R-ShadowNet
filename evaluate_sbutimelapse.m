%% compute rmse|计算RMSE 
% 1`modify the directories 2`run|修改路径,再运行
clear;close all;clc

file_folder='C:\资料\2020-12-DR\sbutimelapse\SBU-Time\frames\';
file_list=dir(file_folder);
size_row = size(file_list);
folder_num = size_row(1);
names={};
for i=3:folder_num
    names = [names,file_list(i,1).name];
end

pppsnrs=zeros(1,length(names)); 
ssssims=zeros(1,length(names));
for iii=1:length(names)

    maskname = ['C:\资料\2020-12-DR\sbutimelapse\SBU-Time\pseudo_anno\',names{iii},'_mask.png'];
    gtname = ['C:\资料\2020-12-DR\sbutimelapse\SBU-Time\pseudo_anno\',names{iii},'_max.png'];
    mask=imread(maskname);
    m=imresize(mask,[256 256]);
    nmask=~m;       %mask of non-shadow regions|非阴影区域的mask
    smask=~nmask;   %mask of shadow regions|阴影区域的mask
    gt=imread(gtname);
    f=imresize(gt,[256 256]);
    f = double(f)/255;
    
%     shadowdir = ['C:\资料\2020-12-DR\sbutimelapse\SBU-Time\frames\',names{iii},'\']; 
%     shadowdir = ['C:\资料\2020-12-DR\sbutimelapse\ours\B_100_mask6_',names{iii},'\']; 
%     shadowdir = ['C:\资料\2020-12-DR\sbutimelapse\lgsn-sbu\B_SBU_',names{iii},'\']; 
    shadowdir = ['C:\资料\2020-12-DR\sbutimelapse\g2r-sbu\B_101_mask6_noreplace_',names{iii},'\']; 

    SD = dir([shadowdir '/*.png']);
    ppsnrs=zeros(1,size(SD,1)); 
    sssims=zeros(1,size(SD,1));
    
    for i=1:size(SD)
        sname = strcat(shadowdir,SD(i).name); 
        s=imread(sname);
        s=imresize(s,[256 256]);
        s = double(s)/255;
        ppsnrs(i)=psnr(s.*smask,f.*smask);
        sssims(i)=ssim(s.*smask,f.*smask);
%         ppsnrs(i)=psnr(s.*repmat(smask,[1 1 3]),f.*repmat(smask,[1 1 3]));
%         sssims(i)=ssim(s.*repmat(smask,[1 1 3]),f.*repmat(smask,[1 1 3]));
    end
    pppsnrs(iii)=mean(ppsnrs);
    ssssims(iii)=mean(sssims);
    disp(names{iii})
    fprintf('%f\t%f\n\n',pppsnrs(iii),ssssims(iii));
end
    fprintf('average PSNR,SSIM:\n%f\t%f\n',mean(pppsnrs),mean(ssssims));
    