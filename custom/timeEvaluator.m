function [tPassed, tRemainingEst, msg] = timeEvaluator(i, n)
    persistent t
    if(isempty(t)) % Ýlk çaðrýldýðýnda t'yi þu anki zamana eþitle
        t = clock;
    end
    tPassed =  etime(clock, t); % Geçen süre = þu anki zaman - bir önceki ölçüm

    if nargout > 1 % Birden fazla sonuç isteniyorsa 
        assert(nargin == 2) % Þu anki döngü numarasý ve toplam döngü sayýsý verilmeli
        tRemainingEst = tPassed / i * (n-i);
        if nargout > 2 % mesaj olarak basýlacaksa
            msg = sprintf('Elapsed time: %.1f sec. Estimated remaining time: %.2f sec\n', tPassed, tRemainingEst);
        end
    end
end