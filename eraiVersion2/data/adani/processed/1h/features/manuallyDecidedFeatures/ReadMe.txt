Attribute difference based features
-----------------------------------------------

Following features have been decided for training the model. SHOULD BE FINE TUNED BASED ON PERFORMANCE

				open		high		low		close
------------------------------------------------------------------------------------
high_low_diff_exp_inv_2		-0.453788	-0.466815	-0.440574	-0.454519
close_high_diff_exp_1		-0.363132	-0.366613	-0.344964	-0.345216
close_low_diff_exp_inv_2	-0.290095	-0.307125	-0.285133	-0.307312
open_low_diff_exp_inv_1		-0.301008	-0.293140	-0.275037	-0.277944
open_high_diff_exp_1		-0.266702	-0.293119	-0.273983	-0.291301


open_close_mid	0.999825	0.999785	0.999748	0.999825
open_high_mid	0.999884	0.999886	0.999508	0.999604
open_low_mid	0.999884	0.999517	0.999882	0.999572
close_high_mid	0.999501	0.999920	0.999515	0.999919
close_low_mid	0.999511	0.999565	0.999903	0.999904
high_low_mid	0.999722	0.999817	0.999811	0.999832

data_magnitude	0.999786	0.999819	0.999785	0.999840

magnitudeTimesScarcity_by_prior_holidays_pow_3	0.273032	0.272661	0.272825	0.272343

green_red_vector_pow_2		0.901418	0.902617	0.900237	0.901416	
red_candle_magnitude_pow_4	0.601855	0.593542	0.594226	0.587961
green_candle_magnitude_pow_4	0.536609	0.546420	0.544062	0.551389

redCandlesBySizeTimesMagnitude_pow_2	0.459723	0.450875	0.443612	0.438564	
greenCandlesBySizeTimesMagnitude_pow_3	0.413663	0.429470	0.420011	0.432960	
redCandlesBySizeTimesMagnitude_pow_3	-0.464503	-0.457274	-0.450526	-0.446462	