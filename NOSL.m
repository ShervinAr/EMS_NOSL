function [a,b,theta] = NOSL(K,reg1,reg2,c2,Res)

n = size(K,1);
a0 = (K+reg1*eye(n,n))\Res;
a = a0;
Y = K*a;

%A = Y*Y';
n = size(Y,1);
E = EuDist2(Y,Y,0);%(eye(n,n).*A)*ones(n,n)+ones(n,n)*(eye(n,n).*A)'-2*A;
theta = c2/mean(E(:));
J = exp(-1*theta*E);
J = .5*(J+J');
b = (J+reg2*eye(n,n))\Res;

%costs(1) = norm(J*b-Res)^2+reg1*trace(a'*K*a)+reg2*trace(b'*J*b);
st = 1;
for i=1:100

    % update a
    JK2 = 2*(J*b-Res)*b';
    JE = (-1*theta*J).*JK2;
    JA = eye(n,n).*((JE+JE')*ones(n,n))-2*JE;
    JY = (JA+JA')*Y;
    Grd = K*JY+reg1*2*K*a;
    if norm(Grd)/size(Grd,1)/size(Grd,2)>1e-5
        Grd = Grd./norm(Grd);%normc(Grd);%
        step = st;
        funcold = norm(J*b-Res)^2+reg1*trace(a'*K*a)+reg2*trace(b'*J*b);
        % step search
        for stepiter = 1:10
            atmp = a-step*Grd;
            Y = K*atmp;
            %A = Y*Y';
            E = EuDist2(Y,Y,0);%(eye(n,n).*A)*ones(n,n)+ones(n,n)*(eye(n,n).*A)'-2*A;
            Jtmp = exp(-1*theta*E);
            btmp = (Jtmp+reg2*eye(n,n))\Res;
            if norm(Jtmp*btmp-Res)^2+reg1*trace(atmp'*K*atmp)+reg2*trace(btmp'*Jtmp*btmp)<(1-1e-8)*funcold
                a = atmp;
                J = Jtmp;
                b = btmp;
                break;
            else
                step = .5*step;
            end
        end
    end

    %     % update theta
    %     JK2 = 2*(J*b-Res)*b';
    %     Jtheta = trace(JK2'*(-1*J.*E));
    %     if abs(Jtheta)>1e-4
    %         Jtheta = Jtheta/norm(Jtheta);
    %         step = st;
    %         funcold = norm(J*b-Res)^2+reg1*trace(a'*K*a)+reg2*trace(b'*J*b);
    %         %step search
    %         for stepiter = 1:10
    %             if theta-step*Jtheta>1e-5
    %                 thetatmp = theta-step*Jtheta;
    %                 Jtmp = exp(-1*thetatmp*E);
    %                 Jtmp = .5*(Jtmp+Jtmp');
    %                 btmp = (Jtmp+reg2*eye(n,n))\Res;
    %                 if norm(Jtmp*btmp-Res)^2+reg1*trace(a'*K*a)+reg2*trace(btmp'*Jtmp*btmp)<(1-1e-8)*funcold
    %                     theta = thetatmp;
    %                     J = Jtmp;
    %                     % update b
    %                     b = btmp;
    %                     break;
    %                 else
    %                     step = .5*step;
    %                 end
    %             else
    %                 step = .5*step;
    %             end
    %         end
    %     end

    %costs(i+1) = norm(J*b-Res)^2+reg1*trace(a'*K*a)+reg2*trace(b'*J*b);
end
% figure,plot(costs)
%
% costs