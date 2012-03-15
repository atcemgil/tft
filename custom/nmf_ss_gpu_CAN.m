
format ('compact')

clear
close all
%clc


addpath('custom/library/miditoolbox');
addpath('custom/library/voicebox');

%% Prepare Instrument Sounds

isolSoundPath = 'custom/data/rwc_excerpt/';

inst1Path = '011PFPEM/';
inst2Path = '172VCNOM/';


frameLen = 4096*2;
numFreq = frameLen/2;
frameInc = round(frameLen/2);
audioWindow = hanning(frameLen);
% audioWindow = ones(frameLen,1);

p = -1; %1:KL divergence
EPOCH = 10;
plotOn = true;

%%%%%%%%
trnFiles1 = dir([isolSoundPath inst1Path '*.wav']);
trnFiles2 = dir([isolSoundPath inst2Path '*.wav']);

I1 = length(trnFiles1);
I2 = length(trnFiles2);

I = I1+I2;

K1= 40;
K2= 20;
K = K1+ K2;


numRepeats = 1;


X3 = zeros(numFreq,35000);
T = zeros(I,35000);


clear smartPrint timeEvaluator

lastTime = 1;

for i=1:I1
    trnFileName = trnFiles1(i).name;
    
    [signal Fs] = wavread([isolSoundPath inst1Path  trnFileName]) ;
    signal = mean(signal,2);
    
    signal = signal(1:5*Fs);
    
    
    sigFrames = enframe(signal,audioWindow,frameInc)';
    Xtemp = abs(fft(sigFrames));
    Xtemp = Xtemp(1:numFreq,:);
    
    if(p == 2)
       Xtemp = Xtemp.^2; 
    end
    
    Ttemp = zeros(I,size(Xtemp,2));
    Ttemp(i,:) = 1;
    
   
    
    X3(:,lastTime:(lastTime+size(Xtemp,2)-1)) = Xtemp;
    T(:, lastTime:(lastTime+size(Ttemp,2)-1)) = Ttemp;
    
    lastTime = lastTime+size(Xtemp,2);
    
    
    
    %                     imagesc(log(Xtemp));set(gca,'ydir','n'); colormap(specol);
    %                     drawnow;
    %                     pause
    fprintf(smartPrint(sprintf('Inst1 file %d is processed. \n',i)));
    
end


clear smartPrint timeEvaluator

for i=1:I2
    trnFileName = trnFiles2(i).name;
    
    [signal Fs] = wavread([isolSoundPath inst2Path  trnFileName]) ;
    signal = mean(signal,2);
    
   
    
    sigFrames = enframe(signal,audioWindow,frameInc)';
    Xtemp = abs(fft(sigFrames));
    Xtemp = Xtemp(1:numFreq,:);
    
    if(p == 2)
       Xtemp = Xtemp.^2; 
    end
    
    Ttemp = zeros(I,size(Xtemp,2));
    Ttemp(I1+i,:) = 1;
    
    
    
    X3(:,lastTime:(lastTime+size(Xtemp,2)-1)) = Xtemp;
    T(:, lastTime:(lastTime+size(Ttemp,2)-1)) = Ttemp;
    
    lastTime = lastTime+size(Xtemp,2);
    
    
    %                     imagesc(log(Xtemp));set(gca,'ydir','n'); colormap(specol);
    %                     drawnow;
    %                     pause
    fprintf(smartPrint(sprintf('Inst2 file %d is processed. \n',i)));
    
end

numTimeFrames_train = lastTime;
X3 = X3(:,1:numTimeFrames_train);
T = T(:,1:numTimeFrames_train);



%% Prepare Test File

testSoundPath = 'custom/data/pianoDuet/';
midiFileName = 'swan';

[signalBase fs] = wavread([testSoundPath midiFileName '.wav']);

signalBase =  signalBase(1:4*fs,:);

mixFactor = 0.7;
signal = mixFactor * signalBase(:,1) + (1-mixFactor)*signalBase(:,2);

sigFrames = enframe(signal,audioWindow,frameInc)';
X1 = abs(fft(sigFrames));
X1Phase = angle(fft(sigFrames));
X1 = X1(1:numFreq,:);
numTimeFrames_test = size(X1,2);

M1 = ones(size(X1));

if(p == 2)
    X1= X1.^2;
end


%% Prepare MIDI

midiScaleFactor = 100;

midiMtx = midi2nmat([testSoundPath midiFileName '.mid']);

chans = midiMtx(:,3);
chans(chans == 3) = 2;
midiMtx(:,3) = chans;


midiMtx1 = midiMtx(midiMtx(:,3) == 2,:);
midiMtx2 = midiMtx(midiMtx(:,3) == 1,:);



X2_1 = midiScaleFactor * prepareMidiGroundTruth_midimtx(midiMtx1,Fs, frameLen,21:108);
X2_2 = midiScaleFactor * prepareMidiGroundTruth_midimtx(midiMtx2,Fs, frameLen,48:93);


X2_1 = X2_1(:,1:numTimeFrames_test);
X2_2 = X2_2(:,1:numTimeFrames_test);

ix = find(sum(X2_1) ~= 0,1,'first');
X2_1 = X2_1(:,ix:end);

ix = find(sum(X2_2) ~= 0,1,'first');
X2_2 = X2_2(:,ix:end);


numTimeFramesMidi = size(X2_1,2) +size(X2_2,2);

X2 = zeros(I,numTimeFramesMidi);
X2(1:I1,1:size(X2_1,2)) = X2_1;
X2(I1+1:I,size(X2_1,2)+1:numTimeFramesMidi) = X2_2;


Z = zeros(I,K);
Z(1:I1,1:K1) = 1;
Z(I1+1:I,K1+1:K) = 1;


Y = zeros(K,numTimeFramesMidi);
Y(1:K1,1:size(X2_1,2)) = 1;
Y(K1+1:K,size(X2_1,2)+1:numTimeFramesMidi) = 1;



%% PREPARE FOR GPU LIB

% DO NOT CHANGE ORDER!!!
V_card_sym=['f','i','k','t','m','n'];
V_cards=[numFreq, I, K, numTimeFrames_test,numTimeFramesMidi,numTimeFrames_train]


R=[ 1, 1, 1, 1, 0, 0, 0, 0 ;  
    0, 1, 1, 0, 1, 1, 0, 0 ;
    1, 0, 0, 0, 0, 0, 1, 1 ];


updateD=1;
updateB=1;
updateZ=0;
updateC=1;
updateG=1;
updateY=0;
updateF=1;
updateT=0;


X1_card_sym=['f','t'];
X2_card_sym=['i','m'];
X3_card_sym=['f','n'];
D_card_sym=['f','i'];
B_card_sym=['i','k'];
Z_card_sym=['i','k'];
C_card_sym=['k','t'];
G_card_sym=['k','m'];
Y_card_sym=['k','m'];
F_card_sym=['i','n'];
T_card_sym=['i','n'];
BC_card_sym=['i','k','t'];

eps = 1e-5;
eps2 = 1e-3;
X1(X1<eps2) = eps2;
X2(X2<eps2) = eps2;
X3(X3<eps2) = eps2;

gX1 = reshape(X1,[numFreq, 1, 1, numTimeFrames_test,1,1]);
gX2 = reshape(X2,[1, I, 1, 1,numTimeFramesMidi,1]);
gX3 = reshape(X3,[numFreq, 1, 1, 1,1,numTimeFrames_train]);
gZ  = reshape(Z,[1, I, K, 1,1,1]);
gY  = reshape(Y,[1, 1, K, 1,numTimeFramesMidi,1]);
gT  = reshape(T,[1, I, 1, 1,1,numTimeFrames_train]);


%% MUR

clear smartPrint 

for nn = 1:numRepeats
    
    %fprintf('\n%d\t%d\t%d\n',nn);
    
    
    
    % MUR
    
    D = 50*rand(numFreq,I);
    B = 50*rand(I,K);
    C = 50*rand(K,numTimeFrames_test);
    G = 50*rand(K, numTimeFramesMidi) ;
    F = 50*rand(I, numTimeFrames_train);
    
    
    gD  = reshape(D,[numFreq, I, 1, 1,1,1]);
    gB  = reshape(B,[1, I, K, 1,1,1]);
    gC  = reshape(C,[1, 1, K, numTimeFrames_test,1,1]);
    gG  = reshape(G,[1, 1, K, 1,numTimeFramesMidi,1]);
    gF  = reshape(F,[1, I, 1, 1,1,numTimeFrames_train]);

    % 'f','i','k','t','m','n'

    maskD1 = zeros(size(D));
    maskD2 = zeros(size(D));

    maskD_card_sym = ['f', 'i'];
    gmaskD1 = reshape(maskD1, [numFreq, I, 1, 1, 1, 1]);
    gmaskD2 = reshape(maskD2, [numFreq, I, 1, 1, 1, 1]);
    
    tic; 
    % DO NOT CHANGE TENSOR ORDER!
    [r_D r_B r_Z r_C r_G r_Y r_F r_T ] = ...
        umut01_par ( 1, ...
                   V_card_sym, ...
                   V_cards, ...
                   p, ...
                   R, ...
                   X1_card_sym, gX1, ...
                   X2_card_sym, gX2, ...
                   X3_card_sym, gX3, ...
                   D_card_sym,  gD, updateD, ...
                   B_card_sym,  gB, updateB, ...
                   Z_card_sym,  gZ, updateZ, ...
                   C_card_sym,  gC, updateC, ...
                   G_card_sym,  gG, updateG, ...
                   Y_card_sym,  gY, updateY, ...
                   F_card_sym,  gF, updateF, ...
                   T_card_sym,  gT, updateT ...
                   );
    toc
    
    
    
    

    
    if(plotOn)
        f1 = figure;
        f2 = figure;
        
    end
    
    
    
    if(plotOn)
        
        set(0,'CurrentFigure',f1)
        subplot(6,6,[3 4]);
        imagesc(BC(1:I1,:)); set(gca,'ydir','n');
        title('B*C piano');
        colorbar;
        
        subplot(6,6,[9 10]);
        imagesc(BC(I1+1:end,:)); set(gca,'ydir','n');
        title('B*C vocal');
        colorbar;
        
        subplot(6,6,[5 6 11 12]);
        imagesc(F.*T); set(gca,'ydir','n');
        title('F');
        colorbar;
        
        subplot(6,6,[27 28 33 34]);
        imagesc(log(M1.*X1)); set(gca,'ydir','n');
        title('X1');
        clr1 = caxis;
        colorbar;
        
        subplot(6,6,[29 30 35 36]);
        imagesc(log(X3)); set(gca,'ydir','n');
        title('X3');
        clr3 = caxis;
        colorbar;
        
        subplot(6,6,[15 16 21 22]);
        imagesc(log(X1hat)); set(gca,'ydir','n');
        title('X1hat');
        caxis(clr1);
        colorbar;
        
        subplot(6,6,[17 18 23 24]);
        imagesc(log(X3hat)); set(gca,'ydir','n');
        title('X3hat');
        caxis(clr3);
        colorbar;
        
        subplot(6,6,[25 31]);
        imagesc(log(D(:,1:I1))); set(gca,'ydir','n');
        title('D piano');
        colorbar;
        
        subplot(6,6,[26 32]);
        imagesc(log(D(:,I1+1:end))); set(gca,'ydir','n');
        title('D vocal');
        colorbar;
        
        
        drawnow;
        
        set(0,'CurrentFigure',f2)
        subplot(5,3,[10 13]);
        imagesc(B.*Z); set(gca,'ydir','n');
        colorbar;
        title('B');
        
        subplot(5,3,[2 3]);
        imagesc(G.*Y); set(gca,'ydir','n');
        colorbar;
        title('G');
        
        
        subplot(5,3,[5 6 8 9]);
        imagesc(X2hat); set(gca,'ydir','n');
        title('X2hat');
        caxis([0 midiScaleFactor]);
        colorbar;
        
        
        subplot(5,3,[11 12 14 15]);
        imagesc(X2); set(gca,'ydir','n');
        title('X2');
        colorbar;
        
        drawnow;
        
        
        
    end
    
    
end





%%



Dsub = D(:,1:I1);
Bsub = B.*Z;
Bsub = Bsub(1:I1,1:K1);
Csub = C(1:K1,:);

thr = 0.1;

BC = Bsub*Csub;
BC(BC<thr*max(BC(:))) = 0;

figure, imagesc(BC); set(gca,'ydir','n'); colorbar

X1piano = Dsub*BC;


Dsub = D(:,I1+1:end);
Bsub = B.*Z;
Bsub = Bsub(I1+1:end,K1+1:end);
Csub = C(K1+1:end,:);

BC = Bsub*Csub;
BC(BC<thr*max(BC(:))) = 0;

figure, imagesc(BC); set(gca,'ydir','n'); colorbar

X1vocal = Dsub*BC;



X1piano = X1.*(X1piano./X1hat);


if(p == 2)
   X1piano = sqrt(X1piano); 
end


X1piano = [X1piano; conj(X1piano(end:-1:1,:))];
sigPiano = X1piano .* exp(sqrt(-1)*X1Phase);
sigPiano = real(ifft(sigPiano));
sigPiano = overlapadd(sigPiano',audioWindow,frameInc);


X1vocal = X1.*(X1vocal./X1hat);

if(p == 2)
   X1vocal = sqrt(X1vocal); 
end


X1vocal = [X1vocal; conj(X1vocal(end:-1:1,:))];
sigVocal = X1vocal .* exp(sqrt(-1)*X1Phase);
sigVocal = real(ifft(sigVocal));
sigVocal = overlapadd(sigVocal',audioWindow,frameInc);

% X1rec = D*(B.*Z)*C;
% 
% X1rec = [X1rec; conj(X1rec(end:-1:1,:))];
% sigRec = X1rec .* exp(sqrt(-1)*X1Phase);
% sigRec = real(ifft(sigRec));
% sigRec = overlapadd(sigRec',audioWindow,frameInc);




