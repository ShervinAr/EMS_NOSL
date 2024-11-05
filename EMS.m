clear
clc

addpath('...\FMRVR\');

% load sample data 
load('Month12_train_input.mat');
Trdata = Month12_train_input;
Month12_train_input = [];
S = eps+quantile(Trdata,.95);
Trdata = (Trdata./S);
load('Month12_train_output.mat');
TrRes = Month12_train_output;
Month12_train_output = [];

% divide into training and validation
X = 0.5; % fraction for validation
rd = 1:size(Trdata,1);
Vddata = Trdata(rd(1:floor(X*size(Trdata,1))),:);
VdRes = TrRes(rd(1:floor(X*size(TrRes,1))),:);
Trdata = Trdata(rd(1+floor(X*size(Trdata,1)):end),:);
TrRes = TrRes(rd(1+floor(X*size(TrRes,1)):end),:);

load('Month12_test_input.mat');
Tedata = Month12_test_input;
Month12_test_input = [];
Tedata = (Tedata./S);
load('Month12_test_output.mat');
TeRes = Month12_test_output;
Month12_test_output = [];

%%
n = size(Trdata,1);
Etr = EuDist2(Trdata,Trdata,0);
Evd = EuDist2(Vddata,Trdata,0);
Md = mean(Etr(:));

%% training
% Nonlinear composition
mn = 1e+20;

for c1 = [.1 .5 1]

    K = exp(-1*c1/Md.*Etr);
    K = .5*(K+K');
    Kvd = exp(-1*c1/Md.*Evd);

    for reg1 = [1e-4 1e-3 1e-2 1e-1]
        for reg2 = [1e-4 1e-3 1e-2 1e-1]

            for c2 = [.1 .5 1]

                [a1,b1,theta1] = NOSL(K,reg1,reg2,c2,TrRes);
                E2 = EuDist2(Kvd*a1,K*a1,0);
                Z1tmp = exp(-1*theta1*E2)*b1;

                if sum(sum((Z1tmp-VdRes).^2))<mn
                    mn = sum(sum((Z1tmp-VdRes).^2));
                    r21Opt = reg1;
                    c21Opt = c1;
                    r22Opt = reg2;
                    c22Opt = c2;
                    a1Opt = a1;
                    b1Opt = b1;
                    theta1Opt = theta1;
                    K1Opt = K;
                    Kvd1Opt = Kvd;
                end
            end
        end
    end
end

% Kernel regression
mn = 1e+20;
for c1 = [.1 .5 1]

    K = exp(-1*c1/Md.*Etr);
    K = .5*(K+K');
    Kvd = exp(-1*c1/Md.*Evd);

    for reg1 = [1e-4 1e-3 1e-2 1e-1]

        a = (K+reg1*eye(n,n))\TrRes;
        Z2tmp = Kvd*a;
        if sum(sum((Z2tmp-VdRes).^2))<mn
            mn = sum(sum((Z2tmp-VdRes).^2));
            r11Opt = reg1;
            c11Opt = c1;
            a2Opt = a;
            K2Opt = K;
            Kvd2Opt = Kvd;
        end
    end
end

% SVR
mn = 1e+20;
for c1 = [.1 .5 1]

    K = exp(-1*c1/Md.*Etr);
    K = .5*(K+K');
    Kvd = exp(-1*c1/Md.*Evd);

    Phi = [ones(size(K,1),1) K];
    [used, alpha, Mu, invSigma, OmegaHat] = fmrvr(Phi,TrRes,100,1e-6);

    Phi = [ones(size(Kvd,1),1) Kvd];
    Z3tmp = Phi(:,used)*Mu;

    if sum(sum((Z3tmp-VdRes).^2))<mn
        mn = sum(sum((Z3tmp-VdRes).^2));
        c13Opt = c1;
        K3Opt = K;
        Kvd3Opt = Kvd;
        usedOpt = used;
        MuOpt = Mu;
    end
end

%% test responses

% Nonlinear composition
Kte = exp(-1*c21Opt/Md*EuDist2(Tedata,Trdata,0));
E2 = EuDist2(Kte*a1Opt,K1Opt*a1Opt,0);
Kte = exp(-1*theta1Opt*E2);
Z1te = Kte*b1Opt;

% Kernel regression
Kte = exp(-1*c11Opt/Md.*EuDist2(Tedata,Trdata,0));
Z2te = Kte*a2Opt;

% SVR
Kte = exp(-1*c13Opt/Md.*EuDist2(Tedata,Trdata,0));
Phi = [ones(size(Kte,1),1) Kte];
Z3te = Phi(:,usedOpt)*MuOpt;


%% training responses

% Nonlinear composition
E2 = EuDist2(K1Opt*a1Opt,K1Opt*a1Opt,0);
Ktr = exp(-1*theta1Opt*E2);
Z1tr1 = Ktr*b1Opt;
E2 = EuDist2(Kvd1Opt*a1Opt,K1Opt*a1Opt,0);
Kvd = exp(-1*theta1Opt*E2);
Z1tr2 = Kvd*b1Opt;
Z1tr = [Z1tr2;Z1tr1];

% Kernel regression
Z2tr1 = K2Opt*a2Opt;
Z2tr2 = Kvd2Opt*a2Opt;
Z2tr = [Z2tr2;Z2tr1];

% SVR
Phi = [ones(size(K3Opt,1),1) K3Opt];
Z3tr1 = Phi(:,usedOpt)*MuOpt;
Phi = [ones(size(Kvd3Opt,1),1) Kvd3Opt];
Z3tr2 = Phi(:,usedOpt)*MuOpt;
Z3tr = [Z3tr2;Z3tr1];



