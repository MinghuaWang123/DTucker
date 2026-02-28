function [ output_args ] = calcDenoiseResult( CleanTensor,NoiseTensor,DenoiseTensor,state ,debug)
%统计去噪后的结果
%  输入分别为干净张量，噪声张量，去噪后张量


[sx,sy,sz]=size(CleanTensor);
noise_psnr=zeros(1,sz);%噪声的psnr
denoise_psnr=zeros(1,sz);%去噪后的psnr
denoise_ssim=zeros(1,sz);%去噪后的SSIM
noise_ssim=zeros(1,sz);
denoise_sam=zeros(1,sz);%去噪后的SAM
for i=1:sz
    noise_psnr(1,i)=psnr(NoiseTensor(:,:,i),CleanTensor(:,:,i));%噪声图像的psnr等级
    
    denoise_psnr(1,i)=psnr(DenoiseTensor(:,:,i),CleanTensor(:,:,i));%去噪后图像的psnr等级
    
    noise_ssim(1,i)=ssim(NoiseTensor(:,:,i),CleanTensor(:,:,i));
    denoise_ssim(1,i)=ssim(DenoiseTensor(:,:,i),CleanTensor(:,:,i));%去噪后的SSIM的等级
    sam1=DenoiseTensor(:,:,i);
    sam2=CleanTensor(:,:,i);
%     denoise_sam(1,i)=hyperSam(sam1(:),sam2(:));%去噪后图像的psnr等级
end
output_args.noise_psnr=noise_psnr;
output_args.denoise_psnr=denoise_psnr;

output_args.noise_ssim=noise_ssim;
output_args.denoise_ssim=denoise_ssim;

% output_args.denoise_sam=denoise_sam;
output_args.state=state;%对程序的一些说明


if(debug == true)

        psnrtitle=[state ' PSNR'];
        figure;
        subplot(121);hold on
        plot(noise_psnr,'r');
        plot(denoise_psnr,'g');
        title(psnrtitle);
        legend('NOISE','DENOISE');

        ssimtitle=[state ' SSIM'];
        subplot(122);hold on
        plot(noise_ssim,'r');
        plot(denoise_ssim,'g');
        title(ssimtitle);
        legend('NOISE','DENOISE');
end
        mpsnrsig=mean(noise_psnr);
        mpsnrdenoise=mean(denoise_psnr);
        mssimsig=mean(noise_ssim);
        mssimdenoise=mean(denoise_ssim);
        
%         msam  =mean(denoise_sam);
        fprintf('\n 噪声图像为%f dB,SSIM为 %f ,去噪后为 %f dB SSIM为 %f \n',mpsnrsig,mssimsig,mpsnrdenoise,mssimdenoise);
end

