function [tPassed, tRemainingEst, msg] = timeEvaluator(i, n)
    persistent t
    if(isempty(t)) % �lk �a�r�ld���nda t'yi �u anki zamana e�itle
        t = clock;
    end
    tPassed =  etime(clock, t); % Ge�en s�re = �u anki zaman - bir �nceki �l��m

    if nargout > 1 % Birden fazla sonu� isteniyorsa 
        assert(nargin == 2) % �u anki d�ng� numaras� ve toplam d�ng� say�s� verilmeli
        tRemainingEst = tPassed / i * (n-i);
        if nargout > 2 % mesaj olarak bas�lacaksa
            msg = sprintf('Elapsed time: %.1f sec. Estimated remaining time: %.2f sec\n', tPassed, tRemainingEst);
        end
    end
end