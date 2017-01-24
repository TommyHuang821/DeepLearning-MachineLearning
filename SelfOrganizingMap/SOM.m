function W=SOM(data,option)
% Self-organizing map 
% data: n x dim
[N inputdim]=size(data);
if nargin<2
    fprintf('use the default for SOM, because do rt set SOM parameter' ); 
    fprintf('5x5 SOM map' ); 
    isplot=0;
    numNode=5;
    r=0.1; % learning rate
    MaxInter=1000; % maximum iteration
    sgm0=1; % sigma for neighbor function
else
    if isfield(option,'isplot');isplot=option.isplot;else  isplot=0; end
    if isfield(option,'numNode');numNode=option.numNode;else  numNode=5; end
    if isfield(option,'LearningRate');r=option.LearningRate;else  r=0.1; end
    if isfield(option,'MaxInter');MaxInter=option.MaxInter;else  MaxInter=1000; end
    if isfield(option,'SigmaNF');sgm0=option.SigmaNF;else  sgm0=1; end
end
 


W=rand(numNode,numNode,inputdim)-0.5;% initial weight

if isplot==1
    figure(1)
    plot(data(:,1),data(:,2),'.b')
    hold on
    plot(W(:,:,1),W(:,:,2),'or')
    plot(W(:,:,1),W(:,:,2),'k','linewidth',2)
    plot(W(:,:,1)',W(:,:,2)','k','linewidth',2)
    hold off
    title('t=0');
    drawnow 
end

rmse=[];
Dist_E=@(x,y) norm(x-y,2); %% Euclidean Distance (rrm-2)
K=@(x,y,alpha,sigma) alpha*exp(-((norm(x-y,2)^2)/(2*sigma^2))); %% Neighborhood function 

t=0;
while (1)
    if t>=MaxInter
        break
    end
    n=r*(1-t/MaxInter);
    sgm=sgm0*(1-t/MaxInter);
    %loop for the N inputs
    err_sum=0;
    for i=1:N
        x=data(i,:)';
        e_rrm=zeros(numNode,numNode);
        for j1=1:numNode
            for j2=1:numNode
                w(:,1)=W(j1,j2,:);
                e_rrm(j1,j2)=Dist_E(x,w);
            end
        end
        min_rrm=min(min(e_rrm));
        [minj1 minj2]=find(e_rrm==min_rrm);
        
        % 
        w(:,1)=W(minj1,minj2,:);
        err = power ( x - w , 2 );
        err1=(sum(err));
        err_sum=err_sum+err1;
  
        j1star= minj1;
        j2star= minj2;
        
        %update the winning neuron
        w(:,1)=W(j1star,j2star,:);
        W(j1star,j2star,:)=w+n*(x-w);
       
        %update the neighbour neurons
        for i1=1:numNode
            for i2=1:numNode
                coe=K([i1,i2],[j1star j2star],n,sgm);
                w(:,1)=W(i1,i2,:);
                W(i1,i2,:)=w+coe*(x-w);
            end
        end
        
    end
    t=t+1;
    rmse(t) = sqrt(err_sum/N);
    fprintf( ' Iteration count = %d, rmse = %f\n', t,rmse(t) );  
    
    
    if isplot==1
        figure(1)
        plot(data(:,1),data(:,2),'.b')
        hold on
        plot(W(:,:,1),W(:,:,2),'or')
        plot(W(:,:,1),W(:,:,2),'k','linewidth',2)
        plot(W(:,:,1)',W(:,:,2)','k','linewidth',2)
        hold off
        title(['t=' num2str(t)]);
        drawnow 
    end
    
end