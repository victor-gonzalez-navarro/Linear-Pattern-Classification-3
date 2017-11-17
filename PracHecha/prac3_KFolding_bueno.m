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


%%
K_folds = 10;
cp = cvpartition(length(Labels_train),'kfold',K_folds);

N=10;
KNN_Errors_Train = zeros(K_folds,N);
KNN_Errors_Train2 = zeros(K_folds,N);



for k0=1:K_folds
    logic_vector_train = training(cp,k0);
    logic_vector_valid = test(cp,k0);
    
    X_train_red = X_train(logic_vector_train==1,:);
    Labels_train_red = Labels_train(logic_vector_train==1);
    
    X_valid = X_train(logic_vector_valid==1,:);
    Labels_valid = Labels_train(logic_vector_valid==1);
    
    for p=1:N
        knnclass = fitcknn(X_train_red,Labels_train_red,'NumNeighbors',p);
        knn_out = predict(knnclass,X_valid);
        knn_Pe_train=sum(Labels_valid ~= knn_out)/length(Labels_valid);
        KNN_Errors_Train(k0,p) = knn_Pe_train;
     
        
        knnclass2 = fitcknn(X_train_red,Labels_train_red,'NumNeighbors',p);
        knn_out2 = predict(knnclass2,X_train_red);
        knn_Pe_train2=sum(Labels_train_red ~= knn_out2)/length(Labels_train_red);
        KNN_Errors_Train2(k0,p) = knn_Pe_train2;
    end
end

figure;
KNN_Errors_Sum = sum(KNN_Errors_Train,1)/10;
plot(KNN_Errors_Sum);

figure;
KNN_Errors_Sum_2 = sum(KNN_Errors_Train2,1)/10;
plot(KNN_Errors_Sum_2);

%% -----------------------------------------------------------------
% en vez de 3 sería mejor poner ls que de un min en KNN_Errors_Sum 
knnclasstest = fitcknn(X_test,Labels_test,'NumNeighbors',5);
knn_out_test = predict(knnclasstest,X_test);
knn_Pe_test=sum(Labels_test ~= knn_out_test)/length(Labels_test);
fprintf(1,' error knn test = %g   \n', knn_Pe_test);
