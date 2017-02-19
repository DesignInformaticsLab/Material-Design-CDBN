k=0;
for j = 1:100
    hidstate=reshape(double(im2bw(reconst_4to3(:,j),0.9)),[373248/288 288])';
    for i = 1:288
        temp=hidstate(i,:);
        if sum(temp(:))>0.1*sum(hidstate(:))
            temp(:)=0;
            k=k+1;
        end
        hidstate_sim(i,:)=temp;
    end
    hidstate_sim_f(:,j)=hidstate_sim(:);
end

% 
% for i = 1:100
% temp=reshape(store_total6(:,i),[144 144]);
% temp=imresize(temp,[168 168]);
% [Gmag,Gdir] = imgradient(temp);
% temp=double(im2bw(temp,0.1));
% temp(Gmag>max(Gmag(:)*0.2639))=1;
% recon_grad4(:,i)=temp(:);
% end

% for ii = 1:100
%     negdata=recon_grad1(:,ii);
%     negdata=reshape(negdata,[168 168]);
% fname = sprintf('recon_5th_gradient_%d',ii);
% save(sprintf('%s.txt',fname),'negdata', '-ascii');
% end