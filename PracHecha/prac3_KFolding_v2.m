% updated to Matlab2016
% MC 2017
close all;
clear;
clc;
%OPCIONES
i_dib=1;					%0 NO /1 SI: DIBUJOS DE DIGITOS
i_CM=1;						%0 NO /1 SI: CALCULA MATRIZ DE CONFUSION
N_classes=10;
%% Elección de la transformada y reducción de dimensión
disp(' ')
disp('Elegir Tipo de Reduccion')
i_red=input(' No reducir (0), PCA (1), MDA(2) =');
if i_red >0
    disp(' ')
    disp('Elegir Dimensión Reducida')
    N_feat=input(' Dim =  ');
else
    N_feat=256;
end
N_dim=16;
%% Lectura BD de train
X_train=[];             % Matriz de Nx256 que contiene todos los vectores
                        % Cada muestra va entre 0(Negro) y 1(Blanco)
Labels_train=[];        % Etiquetas (Inicialmente los datos estan ordenados
                        % por clases del 0 al 9)
for k=0:N_classes-1
    nombre=sprintf('train%d.txt',k);  
    [data] = textread(nombre,'','delimiter',',');
    %data=round(data);  %OPCIONAL elimina los grises
                            %y lo deja todo en blanco y negro
    X_train=[X_train;data];
    N_size=size(data);
    Labels_train=[Labels_train;k*ones(N_size(1),1)];
end
clear nombre data k N_size

%% Lectura BD de test
nombre=sprintf('zip.test');
[data] = textread(nombre,'','delimiter',' ');
Labels_test =data(:,1);
X_test=data(:,2:size(data,2));
clear nombre data
%% OPCION TRANSFORMADAS
if i_red >0
    
    if i_red==1 %PCA
        M_train=mean(X_train);
        X_train2=X_train-ones(length(Labels_train),1)*M_train;
        COEFF = pca(X_train2);
        W=COEFF(:,1:N_feat);
        X_train=X_train2*W;
        X_test=(X_test-ones(length(Labels_test),1)*M_train)*W;
    elseif i_red==2 %MDA
        COEFF= mda_clp(X_train,Labels_train+1,N_classes);
        W=COEFF(:,1:N_feat);
        X_train=X_train*W;
        X_test=X_test*W;
    end

end

%% Create a knn classifier:

KNN_Clusters = 10;
KNN_Members_Cluster = floor(size(X_train,1)/KNN_Clusters);
KNN_Clusters_Last = KNN_Clusters*KNN_Members_Cluster;
KNN_Clusters_Position = 1:KNN_Members_Cluster:KNN_Clusters_Last;

KNN_Errors_Train = zeros(KNN_Clusters,10);
KNN_Errors_Train_2 = zeros(KNN_Clusters,10);


for KNN_Cluster=1:KNN_Clusters

    KNN_Cluster_Validation = X_train(KNN_Clusters_Position(KNN_Cluster):KNN_Clusters_Position(KNN_Cluster)+KNN_Members_Cluster,:);
    KNN_Cluster_Validation_Labels = Labels_train(KNN_Clusters_Position(KNN_Cluster):KNN_Clusters_Position(KNN_Cluster)+KNN_Members_Cluster,:);
    
    KNN_Cluster_Train = X_train(:,:);
    KNN_Cluster_Train(KNN_Clusters_Position(KNN_Cluster):KNN_Clusters_Position(KNN_Cluster)+KNN_Members_Cluster,:) = [];
    KNN_Cluster_Train_Labels = Labels_train;
    KNN_Cluster_Train_Labels(KNN_Clusters_Position(KNN_Cluster):KNN_Clusters_Position(KNN_Cluster)+KNN_Members_Cluster,:) = [];
        
    for K_neig=1:10

        knnclass = fitcknn(KNN_Cluster_Train,KNN_Cluster_Train_Labels,'NumNeighbors',K_neig);
        knn_out = predict(knnclass,KNN_Cluster_Validation);
        knn_Pe_train=sum(KNN_Cluster_Validation_Labels ~= knn_out)/length(KNN_Cluster_Validation_Labels);
        KNN_Errors_Train(KNN_Cluster,K_neig) = knn_Pe_train;
        
        
      % -----------------------------------------------------------------
  
        knn_out_2 = predict(knnclass,KNN_Cluster_Train);
        knn_Pe_train_2=sum(KNN_Cluster_Train_Labels ~= knn_out_2)/length(KNN_Cluster_Train_Labels);
        KNN_Errors_Train_2(KNN_Cluster,K_neig) = knn_Pe_train_2;
        
      % -----------------------------------------------------------------
  
    end

end

figure;
KNN_Errors_Sum = sum(KNN_Errors_Train,1)/10;
plot(KNN_Errors_Sum);

figure;
KNN_Errors_Sum_2 = sum(KNN_Errors_Train_2,1)/10;
plot(KNN_Errors_Sum_2);


%% -----------------------------------------------------------------
% en vez de 3 sería mejor poner ls que de un min en KNN_Errors_Sum 
knnclasstest = fitcknn(X_test,Labels_test,'NumNeighbors',3);
knn_out_test = predict(knnclasstest,X_test);
knn_Pe_test=sum(Labels_test ~= knn_out_test)/length(Labels_test);
fprintf(1,' error knn test = %g   \n', knn_Pe_test);
% -----------------------------------------------------------------
