% insert num at ind in mat
function data=insert(mat,ind,num)
n=length(mat);
data(ind)=num;
data(1:ind-1)=mat(1:ind-1);
data(ind+1:n+1)=mat(ind:n);
end