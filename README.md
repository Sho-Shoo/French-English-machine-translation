# French to English Machine Translation 

French-to-English machine translator using transformer architecture. Implemented several transformer models using encoder block(s), decoder block(s), and multi-head attention layer. Evaluated translation results using BLEU score. 

## Sample translation results 

Original French is:       je ne suis plus un d butant .

Corresponding English is: i m no longer a rookie .

Translation is:           i m not a beginner anymore .

------------------------------------------------
Original French is:       tout le monde parle de lui en bien .

Corresponding English is: he is well spoken of by everybody .

Translation is:           he is fluent in love with him .

------------------------------------------------
Original French is:       elle est inqui te pour sa s curit .

Corresponding English is: she is anxious about her safety .

Translation is:           she s worried about her safety .

------------------------------------------------
Original French is:       il part pour tokyo demain .

Corresponding English is: he s leaving for tokyo tomorrow .

Translation is:           he is leaving for tokyo tomorrow .

------------------------------------------------
Original French is:       vous tes tir e d affaire .

Corresponding English is: you re off the hook .

Translation is:           you re off the hook .

------------------------------------------------
Original French is:       c est un g nie des math matiques .

Corresponding English is: he is a mathematical genius .

Translation is:           he is a genius in mathematics .

------------------------------------------------
Original French is:       vous tes puis es .

Corresponding English is: you re exhausted .

Translation is:           you re exhausted .

------------------------------------------------
Original French is:       je ne suis pas votre servante .

Corresponding English is: i m not your maid .

Translation is:           i m not your maid .

------------------------------------------------

## BLEU evaluation 

Best performing model achieved the following metrics: 

| Metric | Score |
|--------|-------|
| BLEU-1 | .7903 |
| BLEU-2 | .6739 |
| BLEU-3 | .5548 |
| BLEU-4 | .4707 |