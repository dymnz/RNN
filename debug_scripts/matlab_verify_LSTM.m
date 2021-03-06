P_O = [
0.54723
0.70072
0.73495
0.67727
0.64031];

E_O = [
0.15411
0.25235
0.30992
0.33832
0.34624];

O = [
0.74070  0.60335  0.80691
0.75366  0.78682  0.80964
0.76309  0.88438  0.78775
0.72918  0.91267  0.70934
0.68551  0.91010  0.59915];
Z = [
0.71516  0.75810  0.48887
0.75183  0.82698  -0.05432
0.80115  0.93692  -0.28520
0.84981  0.97549  -0.45986
0.88360  0.98665  -0.54767];  
Z_ = [
0.89766  0.99173  0.53458
0.97714  1.17852  -0.05438
1.10182  1.71224  -0.29334
1.25546  2.19468  -0.49713
1.39197  2.50138  -0.61505];    
F = [
0.58626  0.80416  0.71507
0.82413  0.75856  0.85582
0.93483  0.81065  0.90452
0.97059  0.85407  0.89442
0.98186  0.88273  0.83853];

F_ = [
0.34855  1.41251  0.92012
1.54458  1.14481  1.78096
2.66332  1.45424  2.24844
3.49656  1.76689  2.13674
3.99160  2.01854  1.64734];

I = [
0.76717  0.69820  0.65783
0.89287  0.78465  0.81276
0.93453  0.89330  0.83686
0.94007  0.94529  0.79696
0.92923  0.96634  0.69791];

I_ =[
1.19238  0.83874  0.65365
2.12038  1.29295  1.46807
2.65848  2.12492  1.63508
2.75284  2.84945  1.36740
2.57488  3.35724  0.83739];

Po = [
0.35926  
0.16651  
0.48652];

V = [
0.07375  
0.50071 
0.38414];

C = [
0.54864  0.52930  0.32160
1.12344  1.05040  0.23107
1.79892  1.68846  -0.02966
2.54490  2.36418  -0.39302
3.31981  3.04038  -0.71179];

O_ = [
1.04961  0.41945  1.43007
1.11823  1.30588  1.44770
1.16971  2.03454  1.31139
0.99047  2.34662  0.89218
0.77922  2.31487  0.40192];


Rz = [
1.492896038337097  -0.896283668417616  -0.839615094866424
0.589100580005488  1.013448152231727  -0.976905441366558
0.150246952264685  -0.799473287444316  0.252457923373421];
Ri = [
1.06032  0.76693  1.79131
1.53850  0.58079  -0.72411
0.96176  0.24800  1.10357];
Rf = [
1.73096  1.28659  -0.21264
-0.85761  1.20825  -0.01530
0.89792  1.26923  1.97311];
Ro = [
0.09602  -0.25888  1.94765
1.16798  1.26007  0.95456
-0.78194  0.89490  1.65412];

t = 5;

dP_O = 2 * (P_O(t) - E_O(t));
dY = V * dP_O';

dP_O
dY

dO = dY .* tanh(C(t, :)') .* dsigmoid(O_(t, :)');
dO

dC = dY .* O(t, :)' .* dtanh(C(t, :)')...      
  + Po .* dO;

dC

dF = dC .* C(t - 1, :)' .* dsigmoid(F_(t, :)');
dF

dI = dC .* Z(t, :)' .* dsigmoid(I_(t, :)');
dI

dZ = dC .* I(t, :)' .* dtanh(Z_(t, :)');
dZ

disp(sprintf('--------------------------------')); %#ok<*DSPS>
%%
t = 4

dP_O = 2 * (P_O(t) - E_O(t));
dY = V * dP_O' + Rz' * dZ + Ri' * dI + Rf' * dF + Ro' * dO;


dP_O
dY


dO = dY .* tanh(C(t, :)') .* dsigmoid(O_(t, :)');
dO
return;

dC = dY .* O(t, :)' .* dtanh(C(t, :)')...      
  + Po .* dO;

dC

dF = dC .* C(t - 1, :)' .* dsigmoid(F_(t, :)');
dF

dI = dC .* Z(t, :)' .* dsigmoid(I_(t, :)');
dI

dZ = dC .* I(t, :)' .* dtanh(Z_(t, :)');
dZ


