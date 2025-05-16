%% compute rmse|计算RMSE 
% 1`modify the directories 2`run|修改路径,再运行
clear;close all;clc

% names = {'slovenkia'};
names = {'chair','tower','plant_2','guy','mountain','simple','slovenkia','table'};
% names = {'chair','tower','plant_1','plant_2','guy','mountain','simple','slovenkia','table'};

alll=zeros(1,length(names)); 
nonshadoww=zeros(1,length(names));
shadoww=zeros(1,length(names));

pppsnrs=zeros(1,length(names)); 
ssssims=zeros(1,length(names));

for iii=1:length(names)
%     maskname = ['C:\资料\2020-12-DR\svresult\',names{iii},'_mask.png'];
%     gtname = ['C:\资料\2020-12-DR\svresult\',names{iii},'_max.png'];
    maskname = ['C:\资料\2020-12-DR\svresult\t80\',names{iii},'_mask.png'];
    gtname = ['C:\资料\2020-12-DR\svresult\t80\',names{iii},'_max.png'];
    mask=imread(maskname);
    m=imresize(mask,[256 256]);
    nmask=~m;       %mask of non-shadow regions|非阴影区域的mask
    smask=~nmask;   %mask of shadow regions|阴影区域的mask
    gt=imread(gtname);
    cform = makecform('srgb2lab');
    f=imresize(gt,[256 256]);
    f = double(f)/255;
    f = applycform(f,cform);
    mask = ones([size(f,1),size(f,2)]);
    sumnmask=sum(nmask(:));
    sumsmask=sum(smask(:));
    summask=sum(mask(:));

    % result directory|结果路径
%     shadowdir = ['C:\Users\Administrator\Desktop\shadow_video\frames_100_\',names{iii},'\']; 
%     shadowdir = ['C:\资料\2020-12-DR\model_irsn_dabr_3pp_abl_edge_d1_r2_area2\B_85_mask6_replace480_',names{iii},'\'];  
%     shadowdir = ['C:\资料\2020-12-DR\svresult\ours\B_101_mask6_replace_',names{iii},'\']; 
%     shadowdir = ['C:\资料\2020-12-DR\area2_dalunwen_svresult\B_85_mask6_',names{iii},'\']; 
%     shadowdir = ['C:\资料\2020-09-G2R\Result-VSRD\msgan-svresult\B_180_mask6_replace_',names{iii},'\']; 
    shadowdir = ['C:\资料\2020-09-G2R\Result-VSRD\lgsn-svresult\B_100_mask6_replace_',names{iii},'\']; 
%     shadowdir = ['C:\Users\Administrator\Desktop\New folder1\B_85_mask6_',names{iii},'\']; 
    
%     shadowdir = ['C:\Users\Administrator\Desktop\svresult\ours+\B_101_mask6_replace_',names{iii},'\']; 
%     shadowdir = ['C:\Users\Administrator\Desktop\svresult\ours_noreplace\B_101_mask6_',names{iii},'\']; 
%     shadowdir = ['C:\Users\Administrator\Desktop\msgan-svresult\B_180_mask6_replace_',names{iii},'\']; 
%     shadowdir = ['C:\Users\Administrator\Desktop\model_istda_shallow_colour_run1\B_100_mask6_replace_',names{iii},'\']; 
    SD = dir([shadowdir '/*.png']);

    all=zeros(1,size(SD,1));
    nonshadow=zeros(1,size(SD,1));
    shadow=zeros(1,size(SD,1));
    % ISTD dataset image size 480*640
    
    for i=1:size(SD)
    %     sname = strcat(shadowdir,'99-2.png'); 
        sname = strcat(shadowdir,SD(i).name); 
        s=imread(sname);
        s=imresize(s,[256 256]);
        s = double(s)/255;
        s = applycform(s,cform);
        %abs all
        absall=abs(f - s);
        all(i)=sum(absall(:))/summask;
        % non-shadow
        distance = absall.* repmat(nmask,[1 1 3]);
        nonshadow(i)=sum(distance(:))/sumnmask;
        % shadow
        distance = absall.* repmat(smask,[1 1 3]);
        shadow(i)=sum(distance(:))/sumsmask;
%         disp(i);
    end
    alll(iii)=mean(all);
    nonshadoww(iii)=mean(nonshadow);
    shadoww(iii)=mean(shadow);
    
    % video,image, or pixel???
    disp(names{iii})
    fprintf('Lab(all,non-shadow,shadow):\n%f\t%f\t%f\n',alll(iii),nonshadoww(iii),shadoww(iii));
end
fprintf('Lab(all,non-shadow,shadow):\n%f\t%f\t%f\n',mean(alll),mean(nonshadoww),mean(shadoww));
    
