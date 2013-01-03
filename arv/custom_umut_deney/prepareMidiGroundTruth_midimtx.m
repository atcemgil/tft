function groundTruth = prepareMidiGroundTruth_midimtx(midiMtx,Fs,Wlen,notes)

%     Fs = 44100;
%     Wlen = 8192;
    Wdur = Wlen/Fs;

    notesRaw = midiMtx(:,[6 7 4]);
    notesRaw(:,2) = notesRaw(:,1) + notesRaw(:,2);

    T = round(max(notesRaw(:,2))/Wdur);
    groundTruth = zeros(length(notes),T);

    for i = 1:size(notesRaw,1)

        frameBegin = floor(notesRaw(i,1) /Wdur);
        frameEnd = floor(notesRaw(i,2) /Wdur);
        framePitch = notesRaw(i,3)-notes(1)+1;
        
        if(frameBegin == 0); frameBegin = 1; end;

        groundTruth(framePitch, frameBegin:frameEnd) = 1;


    end
    
end