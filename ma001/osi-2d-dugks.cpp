/*---------this code is the DUGKS  with the Shokhov model for the cavity flow  ------------------------------------------*/
// this code is develop for the comparison as the final code
#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include <time.h>
#include <omp.h>
#include <fstream>
#include <iostream>
using namespace std;
/*------------------set the velocity-----------------------*/
// vv=1 GH 28*28
// vv=2 NL 50*50
// sp=0 no-limiter
// sp=1 limiter
#define VV  2   //chose the velocity sets
#define sp 0  // if use the limiter for slop 

#if VV==1
const int Qx=28;        // discrete velocity in x-direction
const int Qy=28;        // discrete velocity in y-direction
#elif VV==2
const int Qx=32;        // discrete velocity in x-direction
const int Qy=32;        // discrete velocity in y-direction
#else
#endif
/*-----------------------------------------------*/
#define NUM_THREADS 4
const int NX=61;       //mesh point in x-direction
const int NY=61;       //mesh point in x-direction
#define IMIN 1
#define IMAX NX
#define JMIN 1
#define JMAX NY
#define NX1 (NX+1)
#define NY1 (NY+1)
#define NX2 (NX+2)
#define NY2 (NY+2)
#define Lx   1.0
#define Ly   1.0
#define sgn(x) ((x)<0.0?-1.0:1.0)
const double R =0.5;
const double Tw=1.0;
const double Omega=0.5;
const double Kn_ref=0.1;
const double Pr=2./3;
const double Ma=0.01;
const double CFL=0.5;
const int K=0;
const int D=2;
double f[NY2][NX2][Qx][Qy],fb[NY2][NX2][Qx][Qy],tpx[Qx],tpy[Qy],ex[Qx],ey[Qy];//f=f~   fb=f-+  cell ceter
double h[NY2][NX2][Qx][Qy],hb[NY2][NX2][Qx][Qy];//sk
double fx[NX1][Qx][Qy],fy[NY1][Qx][Qy],fin[NX1][Qx][Qy];
double hx[NX1][Qx][Qy],hy[NY1][Qx][Qy],hin[NX1][Qx][Qy];//sk
double rho[NY2][NX2],u[NY2][NX2],v[NY2][NX2],te[NY2][NX2],ua[NY2][NX2],va[NY2][NX2],pxy[NY1][NX1];
double gfx[NY2][NX2][Qx][Qy],gfy[NY2][NX2][Qx][Qy],ghx[NY2][NX2][Qx][Qy],ghy[NY2][NX2][Qx][Qy];//sk
double Co_X[Qy],Co_W[Qy];
void initial(void),flux_x(int j),flux_y(int i);
void macro(void),datadeal(void),boundary(void),evol(void);
void setVelocity(void),slop(void),TECPLOT(void),mesh(void),average_stress(void);
double feq(int kx,int ky, double RHO, double U, double V,double TT);
void Shokhov(int kx,int ky, double RHO, double U, double V,double TT,double qx,double qy);
double dt,tau,hh,cmax,PI;
double Cv,Gamma,vis_ref,vis,fs,hs,U0,UU;
double xx[NX2],yy[NY2],x[NX2],y[NY2]; // x,y  cell center location,   xx,yy  the mesh size
double pmax,xmin,umax,st;
int kk,TS;
int main()
{
    int i,j,m,k,nn,SS;
    int mmax,TEND,goon;
    double err,Erra,Errb,errp,pmax_old;
    double start,finish,duration;
    double tt;
    FILE *fp;
    omp_set_num_threads(NUM_THREADS);
    mmax=TEND=0;
    PI=4.0*atan(1.);
    mesh();	
    initial();
    tt=CFL*xmin/cmax;
    if(PI/tt/10000>1)
        nn=(int)(PI/tt/10000)*10000;
    else
        nn=(int)(PI/tt/1000)*1000;
    dt=PI/nn;
    start=clock();
    for(k=1;k<=20;k++)
    {
        st=k;
        TS=(int)(2*nn/st);// the period
        SS=(int)(TS/50);// the time step to compute stress
        printf("Kn=%.2f st=%.2f NX=%d NY=%d Qx=%d dt=%.2e T=%d SS=%d nn=%d\n",Kn_ref,st,IMAX,JMAX,Qx,tt,TS,SS,nn);
        hh=dt/2;

        TEND=100000000;
        m=0;
        umax=0.;
        pmax=0.;
        pmax_old=1.0;
        err=1.;
        errp=1.0;
        while((m<TEND)&&err>1.0e-5)//convergence?
        {
            U0=UU*cos(st*m*dt);
            evol();
            if(m%SS==0)// each SS step 
            {

                kk=m;
                average_stress();
                //   TECPLOT();
            }
            if(m%TS==0)
            {
                Erra=0.0;Errb=0.0;
                for(i=IMIN;i<=IMAX;i++)
                    for(j=JMIN;j<=JMAX;j++)
                    {
                        Erra+=(fabs(u[j][i]-ua[j][i])+fabs(v[j][i]-va[j][i]));
                        Errb+=fabs(u[j][i])+fabs(v[j][i]);
                        ua[j][i]=u[j][i];va[j][i]=v[j][i];
                    }
                err=Erra/Errb;
                errp=fabs(pmax-pmax_old)/pmax_old;
                if(errp<err)err=errp;
                printf("st=%.2f m=%d time=%.2f err=%.4e pmax=%.6e umax=%.4e\n",st,m,dt*m*st/PI/2,err,pmax,umax/UU);

                pmax_old=pmax;

            }
            m++;
        } 
        fp=fopen("max_evo_kn01_ma001.dat","a");
        fprintf(fp,"%4f %.8e %.8e\n",st,pmax,umax/UU);  
        fclose(fp);
    }
    finish=clock();

    duration = (double)(finish - start) / CLOCKS_PER_SEC;
    printf( "CPU time=%2.2f seconds\n", duration/NUM_THREADS );

    fp=fopen("info.dat","w");
    fprintf(fp,"%d %e \n",m, duration);  
    fclose(fp);

    return 0;
}

void average_stress()
{

    int j,i,kx,ky;
    double sxy,apxy,cx,cy,feqh;
    FILE *fp;
    if(kk%TS==0)
    {
        pmax=0.;
        umax=0.;
    }
    j=JMAX;
    apxy=0.;
    for(i=IMIN;i<=IMAX;i++)
    {
        vis=vis_ref*exp(Omega*log(te[j][i]/Tw));
        tau=vis/R/te[j][i]/rho[j][i];

        sxy=0.;
        for(kx=0;kx<Qx;kx++)
            for(ky=0;ky<Qy;ky++)
            { 

                cx=ex[kx]-u[j][i];cy=ey[ky]-v[j][i];
                feqh=feq(kx,ky,rho[j][i],u[j][i],v[j][i],te[j][i]);
                sxy+=cx*cy*(f[j][i][kx][ky]-feqh);
            }


        sxy=sxy*2*tau/(2*tau+dt);	
        sxy/=(R*Tw*UU);
        apxy+=sxy*xx[i]/Lx;           //compute the average shear stress 
    }
    if(fabs(apxy)>pmax)pmax=fabs(apxy);

    if(fabs(u[JMAX][IMAX/2+1])>umax)umax=fabs(u[JMAX][IMAX/2+1]);


    fp=fopen("apxy_evo.dat","a");
    fprintf(fp,"%.2f %d %.4f %.4e %.4e\n",st,kk,st*dt*kk/(2*PI),apxy,u[JMAX][IMAX/2+1]/UU);  
    fclose(fp);

}

void TECPLOT()
{
    int j,i,kx,ky;
    double xi,yj;
    double qhx,qhy,cx,cy,feqh,sxy,as;
    double vis,tau;
    FILE *fp;
    ofstream fout;
    char filename[25];

    /*
       for(j=JMIN;j<=JMAX;j++)
       for(i=IMIN;i<=IMAX;i++)
       {
       vis=vis_ref*exp(Omega*log(te[j][i]/Tw));
       tau=vis/R/te[j][i]/rho[j][i];

       sxy=0.;
       for(kx=0;kx<Qx;kx++)
       for(ky=0;ky<Qy;ky++)
       { 

       cx=ex[kx]-u[j][i];cy=ey[ky]-v[j][i];
       feqh=feq(kx,ky,rho[j][i],u[j][i],v[j][i],te[j][i]);
       sxy+=cx*cy*(f[j][i][kx][ky]-feqh);
       }


       pxy[j][i]=sxy*2*tau/(2*tau+dt);	
       pxy[j][i]/=(R*Tw*UU);
       }
       as=0.;
       j=JMAX;
       for(i=IMIN;i<=IMAX;i++)
       {
       as+=pxy[j][i]*xx[i]/Lx;
       }
       fp=fopen("evo.dat","a");
       fprintf(fp,"%d %.4e %.8e %.8e %.8e %.8e\n",kk, st*dt*kk/(2*PI),as,u[JMAX][IMAX/2+1]/UU,v[JMAX/2+1][IMIN]/UU,pxy[JMAX][IMAX/2+1]);  
       fclose(fp);


*/
    sprintf(filename,"%s%.6d%s","tec",kk,".dat");
    fout.open(filename,ios::out);
    fout<<"Title=\"Cavity Flow\""<<endl;
    fout<<"VARIABLES=\"X\",\"Y\",\"U\",\"V\",\"TE\",\"wqx\",\"wqy\",\"pressure\""<<endl;
    fout<<"ZONE T=\"BOX-kn01\","<<"I="<<IMAX<<",J="<<JMAX<<",F=POINT"<<endl;
    for(j=JMIN;j<=JMAX;j++)
        for(i=IMIN;i<=IMAX;i++)
        {
            vis=vis_ref*exp(Omega*log(te[j][i]/Tw));
            tau=vis/R/te[j][i]/rho[j][i];

            qhx=0.;
            qhy=0.;
            for(kx=0;kx<Qx;kx++)
                for(ky=0;ky<Qy;ky++)
                { 

                    cx=ex[kx]-u[j][i];cy=ey[ky]-v[j][i];
                    qhx+=0.5*cx*((cx*cx+cy*cy)*f[j][i][kx][ky]+h[j][i][kx][ky]);
                    qhy+=0.5*cy*((cx*cx+cy*cy)*f[j][i][kx][ky]+h[j][i][kx][ky]);

                }

            qhx=qhx*2*tau/(2*tau+dt*Pr);
            qhy=qhy*2*tau/(2*tau+dt*Pr);	
            fout<< x[i] <<" "<<y[j] <<" "<<u[j][i]/UU<<" "<<v[j][i]/UU<<" "<<te[j][i]/Tw<<" "<<qhx/(R*Tw*UU)<<" "<<qhy/(R*Tw*UU)<<" "<<rho[j][i]*te[j][i]<<" "<<endl;
        }
    fout.close();


}

void initial()
{
    int j,i,kx,ky;
    double alpha_ref,omega_ref;
    double Re;
    setVelocity();
    Gamma=5./3; 
    Cv=0.5*(3+K)*R;
    UU=Ma*sqrt(Gamma*R*Tw);
    omega_ref=0.5;
    alpha_ref=1.0;


    //vis_ref=5*(alpha_ref+1)*(alpha_ref+2)*sqrt(PI)/(4*alpha_ref*(5-2*omega_ref)*(7-2*omega_ref))*Kn_ref;
    vis_ref=Kn_ref*Lx*sqrt(2*R*Tw/PI);

    Re=Ly*UU/vis_ref;
    printf("vis_ref=%4e Re=%2e \n",vis_ref,Re);

    for(i=IMIN;i<=IMAX;i++)
        for(j=JMIN;j<=JMAX;j++)
        {
            rho[j][i]=1.0;
            u[j][i]=0.0;
            v[j][i]=0.0;
            te[j][i]=Tw;
        }

    for(i=IMIN;i<=IMAX;i++)
        for(j=JMIN;j<=JMAX;j++)
            for(kx=0;kx<Qx;kx++)
                for(ky=0;ky<Qy;ky++)
                {  
                    f[j][i][kx][ky]=feq(kx,ky,rho[j][i],u[j][i],v[j][i],te[j][i]);
                    h[j][i][kx][ky]=(K+3-D)*R*te[j][i]*f[j][i][kx][ky];
                }

}
void boundary()
{  
    int j,i,kx,ky;
    double w1,w2,RW;
    double rh,uh,vh,th,qhx,qhy;
    double cx,cy;
    double ss;


#pragma omp parallel for private(i,kx,ky,qhx,qhy,cx,cy,fs,hs,vis,tau,w1,w2)
    for(j=JMIN;j<=JMAX;j++)
        for(i=IMIN;i<=IMAX;i++)
        {
            vis=vis_ref*exp(Omega*log(te[j][i]/Tw));
            tau=vis/R/te[j][i]/rho[j][i];
            w1=(2*tau-hh)/(2*tau+dt);
            w2=3*hh/(2*tau+dt);
            qhx=qhy=0.;
            for(kx=0;kx<Qx;kx++)
                for(ky=0;ky<Qy;ky++)
                {
                    cx=ex[kx]-u[j][i];cy=ey[ky]-v[j][i];
                    qhx+=0.5*cx*((cx*cx+cy*cy)*f[j][i][kx][ky]+h[j][i][kx][ky]);
                    qhy+=0.5*cy*((cx*cx+cy*cy)*f[j][i][kx][ky]+h[j][i][kx][ky]);
                }

            qhx=qhx*2*tau/(2*tau+dt*Pr);
            qhy=qhy*2*tau/(2*tau+dt*Pr);			
            for(kx=0;kx<Qx;kx++)
                for(ky=0;ky<Qy;ky++)
                {
                    Shokhov(kx,ky,rho[j][i],u[j][i],v[j][i],te[j][i],qhx,qhy);
                    fb[j][i][kx][ky]=w1*f[j][i][kx][ky]+w2*fs;
                    hb[j][i][kx][ky]=w1*h[j][i][kx][ky]+w2*hs;
                }
        }
#pragma omp parallel for private(kx,ky,ss)
    for(i=IMIN;i<=IMAX;i++)for(kx=0;kx<Qx;kx++)for(ky=0;ky<Qy;ky++)	   
    {
        ss=(y[JMIN+1]-y[JMIN-1])/(y[JMIN+1]-y[JMIN]);
        fb[JMIN-1][i][kx][ky]=ss*fb[JMIN][i][kx][ky]+(1-ss)*fb[JMIN+1][i][kx][ky];
        hb[JMIN-1][i][kx][ky]=ss*hb[JMIN][i][kx][ky]+(1-ss)*hb[JMIN+1][i][kx][ky];

        ss=(y[JMAX+1]-y[JMAX-1])/(y[JMAX]-y[JMAX-1]);
        fb[JMAX+1][i][kx][ky]=ss*fb[JMAX][i][kx][ky]+(1-ss)*fb[JMAX-1][i][kx][ky];
        hb[JMAX+1][i][kx][ky]=ss*hb[JMAX][i][kx][ky]+(1-ss)*hb[JMAX-1][i][kx][ky];
    }
#pragma omp parallel for private(kx,ky,ss)
    for(j=JMIN-1;j<=JMAX+1;j++)for(kx=0;kx<Qx;kx++)for(ky=0;ky<Qy;ky++)
    {
        ss=(x[IMIN+1]-x[IMIN-1])/(x[IMIN+1]-x[IMIN]);
        fb[j][IMIN-1][kx][ky]=ss*fb[j][IMIN][kx][ky]+(1-ss)*fb[j][IMIN+1][kx][ky];
        hb[j][IMIN-1][kx][ky]=ss*hb[j][IMIN][kx][ky]+(1-ss)*hb[j][IMIN+1][kx][ky];

        ss=(x[IMAX+1]-x[IMAX-1])/(x[IMAX]-x[IMAX-1]);
        fb[j][IMAX+1][kx][ky]=ss*fb[j][IMAX][kx][ky]+(1-ss)*fb[j][IMAX-1][kx][ky];
        hb[j][IMAX+1][kx][ky]=ss*hb[j][IMAX][kx][ky]+(1-ss)*hb[j][IMAX-1][kx][ky];
    }
}
void flux_x(int j)
{
    int i,kx,ky;
    double w1,w2;
    double rh,uh,vh,th;
    double aa,bb,rhow,delta;
    double qhx,qhy,cx,cy;
    double gradL,gradR;
#pragma omp parallel for private(kx,ky,gradL,gradR,delta)
    for(i=IMIN-1;i<=IMAX;i++)
        for(kx=0;kx<Qx;kx++)
            for(ky=0;ky<Qy;ky++)
            {  
                delta=0.5*(1+sgn(ex[kx]));

                gradL=(0.5*xx[i]-hh*ex[kx])*gfx[j][i][kx][ky]+(-hh*ey[ky])*gfy[j][i][kx][ky];
                gradR=(-0.5*xx[i+1]-hh*ex[kx])*gfx[j][i+1][kx][ky]+(-hh*ey[ky])*gfy[j][i+1][kx][ky];
                fin[i][kx][ky]= delta*(fb[j][i][kx][ky]+gradL)+(1-delta)*(fb[j][i+1][kx][ky]+gradR);

                gradL=(0.5*xx[i]-hh*ex[kx])*ghx[j][i][kx][ky]+(-hh*ey[ky])*ghy[j][i][kx][ky];
                gradR=(-0.5*xx[i+1]-hh*ex[kx])*ghx[j][i+1][kx][ky]+(-hh*ey[ky])*ghy[j][i+1][kx][ky];
                hin[i][kx][ky]= delta*(hb[j][i][kx][ky]+gradL)+(1-delta)*(hb[j][i+1][kx][ky]+gradR);
            }
    i=IMIN-1;  //left wall
    aa=bb=0.;
    for(kx=0;kx<Qx;kx++)
        if(ex[kx]<0)
        {  
            for(ky=0;ky<Qy;ky++)
            {
                aa+=ex[kx]*fin[i][kx][ky];
                bb-=ex[kx]*feq(kx,ky,1.,0.,0.,Tw);
            }
        }
    rhow=-aa/bb;

    for(kx=0;kx<Qx;kx++)
        if(ex[kx]>0)
        {
            for(ky=0;ky<Qy;ky++)
            {
                fin[i][kx][ky]=feq(kx,ky,rhow,0.,0.,Tw);
                hin[i][kx][ky]=fin[i][kx][ky]*(K+3-D)*R*Tw;
            }
        }
    i=IMAX;  //left wall
    aa=bb=0.;
    for(kx=0;kx<Qx;kx++)
        if(ex[kx]>0)
        {  
            for(ky=0;ky<Qy;ky++)
            {
                aa+=ex[kx]*fin[i][kx][ky];
                bb-=ex[kx]*feq(kx,ky,1.,0.,0.,Tw);
            }
        }
    rhow=-aa/bb;

    for(kx=0;kx<Qx;kx++)
        if(ex[kx]<0)
        {
            for(ky=0;ky<Qy;ky++) 
            {
                fin[i][kx][ky]=feq(kx,ky,rhow,0.,0.,Tw);
                hin[i][kx][ky]=fin[i][kx][ky]*(K+3-D)*R*Tw;
            }
        }
#pragma omp parallel for private(kx,ky,rh,uh,vh,th,qhx,qhy,cx,cy,fs,hs,vis,tau,w1,w2)
    for(i=IMIN-1;i<=IMAX;i++)
    {   
        rh=0.0;uh=0.0;vh=0.0;th=0.0;qhx=0.;qhy=0.;
        for(kx=0;kx<Qx;kx++)
            for(ky=0;ky<Qy;ky++)
            {
                rh+=fin[i][kx][ky];
                uh+=fin[i][kx][ky]*ex[kx];
                vh+=fin[i][kx][ky]*ey[ky];
                th+=(ex[kx]*ex[kx]+ey[ky]*ey[ky])*fin[i][kx][ky]+hin[i][kx][ky];

            }
        uh=uh/rh;
        vh=vh/rh;
        th=0.5*th/rh;
        th=(th-0.5*(uh*uh+vh*vh))/Cv;
        vis=vis_ref*exp(Omega*log(th/Tw));
        tau=vis/R/th/rh;
        w1=(2*tau)/(2*tau+hh);
        w2=hh/(2*tau+hh);
        for(kx=0;kx<Qx;kx++)
            for(ky=0;ky<Qy;ky++)
            {
                cx=ex[kx]-uh;cy=ey[ky]-vh;
                qhx+=0.5*cx*((cx*cx+cy*cy)*fin[i][kx][ky]+hin[i][kx][ky]);
                qhy+=0.5*cy*((cx*cx+cy*cy)*fin[i][kx][ky]+hin[i][kx][ky]);
            }
        qhx=qhx*2*tau/(2*tau+hh*Pr);
        qhy=qhy*2*tau/(2*tau+hh*Pr);			

        for(kx=0;kx<Qx;kx++)
            for(ky=0;ky<Qy;ky++)
            {
                Shokhov(kx,ky,rh,uh,vh,th,qhx,qhy);	
                fx[i][kx][ky]=w1*fin[i][kx][ky]+w2*fs;
                hx[i][kx][ky]=w1*hin[i][kx][ky]+w2*hs;
            }
    }

}

void flux_y(int i)
{
    int j,kx,ky;
    double w1,w2;
    double rh,uh,vh,th,rhow,aa,bb;
    double qhx,qhy,cx,cy;
    double gradD,gradU,delta;

#pragma omp parallel for private(kx,ky,gradD,gradU,delta)
    for(j=JMIN-1;j<=JMAX;j++)
        for(kx=0;kx<Qx;kx++)
            for(ky=0;ky<Qy;ky++)
            {  

                delta=0.5*(1+sgn(ey[ky]));


                gradD=(-hh*ex[kx])*gfx[j][i][kx][ky]+(0.5*yy[j]-hh*ey[ky])*gfy[j][i][kx][ky];
                gradU=(-hh*ex[kx])*gfx[j+1][i][kx][ky]+(-0.5*yy[j+1]-hh*ey[ky])*gfy[j+1][i][kx][ky];
                fin[j][kx][ky]= delta*(fb[j][i][kx][ky]+gradD)+(1-delta)*(fb[j+1][i][kx][ky]+gradU);

                gradD=(-hh*ex[kx])*ghx[j][i][kx][ky]+(0.5*yy[j]-hh*ey[ky])*ghy[j][i][kx][ky];
                gradU=(-hh*ex[kx])*ghx[j+1][i][kx][ky]+(-0.5*yy[j+1]-hh*ey[ky])*ghy[j+1][i][kx][ky];
                hin[j][kx][ky]= delta*(hb[j][i][kx][ky]+gradD)+(1-delta)*(hb[j+1][i][kx][ky]+gradU);


            }
    j=JMIN-1;
    aa=bb=0.;
    for(ky=0;ky<Qy;ky++)
        if(ey[ky]<0)
        {  
            for(kx=0;kx<Qx;kx++)
            {
                aa+=ey[ky]*fin[j][kx][ky];
                bb-=ey[ky]*feq(kx,ky,1.,0.,0.,Tw);
            }
        }
    rhow=-aa/bb;

    for(ky=0;ky<Qy;ky++)
        if(ey[ky]>0)
        {
            for(kx=0;kx<Qx;kx++)
            {
                fin[j][kx][ky]=feq(kx,ky,rhow,0.,0.,Tw);
                hin[j][kx][ky]=fin[j][kx][ky]*(K+3-D)*R*Tw;
            }
        }	
    j=JMAX;
    aa=bb=0.;
    for(ky=0;ky<Qy;ky++)
        if(ey[ky]>0)
        {  
            for(kx=0;kx<Qx;kx++)
            {
                aa+=ey[ky]*fin[j][kx][ky];
                bb-=ey[ky]*feq(kx,ky,1.,U0,0.,Tw);
            }
        }
    rhow=-aa/bb;

    for(ky=0;ky<Qy;ky++)
        if(ey[ky]<0)
        {
            for(kx=0;kx<Qx;kx++)
            {
                fin[j][kx][ky]=feq(kx,ky,rhow,U0,0.,Tw);
                hin[j][kx][ky]=fin[j][kx][ky]*(K+3-D)*R*Tw;
            }
        }	
#pragma omp parallel for private(kx,ky,rh,uh,vh,th,qhx,qhy,cx,cy,fs,hs,vis,tau,w1,w2)
    for(j=JMIN-1;j<=JMAX;j++)
    { 

        rh=0.0;uh=0.0;vh=0.0;th=0.0;qhx=0.;qhy=0.;
        for(kx=0;kx<Qx;kx++)
            for(ky=0;ky<Qy;ky++)
            {
                rh+=fin[j][kx][ky];
                uh+=fin[j][kx][ky]*ex[kx];//
                vh+=fin[j][kx][ky]*ey[ky];//
                th+=((ex[kx]*ex[kx]+ey[ky]*ey[ky])*fin[j][kx][ky]+hin[j][kx][ky]);
            }

        uh=uh/rh;
        vh=vh/rh;
        th=0.5*th/rh;
        th=(th-0.5*(uh*uh+vh*vh))/Cv;
        vis=vis_ref*exp(Omega*log(th/Tw));
        tau=vis/R/th/rh;
        w1=(2*tau)/(2*tau+hh);
        w2=hh/(2*tau+hh);
        for(kx=0;kx<Qx;kx++)
            for(ky=0;ky<Qy;ky++)
            {
                cx=ex[kx]-uh;cy=ey[ky]-vh;
                qhx+=cx*((cx*cx+cy*cy)*fin[j][kx][ky]+hin[j][kx][ky]);
                qhy+=cy*((cx*cx+cy*cy)*fin[j][kx][ky]+hin[j][kx][ky]);

            }


        qhx=0.5*qhx*2*tau/(2*tau+hh*Pr);
        qhy=0.5*qhy*2*tau/(2*tau+hh*Pr);			

        for(kx=0;kx<Qx;kx++)
            for(ky=0;ky<Qy;ky++)
            {
                Shokhov(kx,ky,rh,uh,vh,th,qhx,qhy);	
                fy[j][kx][ky]=w1*fin[j][kx][ky]+w2*fs;
                hy[j][kx][ky]=w1*hin[j][kx][ky]+w2*hs;
            }	

    } 

} 

void evol()
{
    int j,i,kx,ky;
    double fa,gradx,grady,Umax,u_local;
    double wx,wy;

    boundary();
    slop();


    /*----flux in x-direction----*/
    for(j=JMIN;j<=JMAX;j++)
    {
        flux_x(j);
#pragma omp parallel for private(kx,ky,fa,gradx,wx)
        for(i=IMIN;i<=IMAX;i++)
        {
            wx=dt/xx[i];
            for(kx=0;kx<Qx;kx++)
                for(ky=0;ky<Qy;ky++)
                {	
                    fa=fb[j][i][kx][ky]*4/3-f[j][i][kx][ky]/3;
                    gradx=ex[kx]*(fx[i][kx][ky]-fx[i-1][kx][ky]);
                    f[j][i][kx][ky]=fa-wx*gradx;		 

                    fa=hb[j][i][kx][ky]*4/3-h[j][i][kx][ky]/3;
                    gradx=ex[kx]*(hx[i][kx][ky]-hx[i-1][kx][ky]);
                    h[j][i][kx][ky]=fa-wx*gradx;
                }
        }
    }
    for(i=IMIN;i<=IMAX;i++)
    { 
        flux_y(i);
#pragma omp parallel for private(kx,ky,fa,grady,wy)
        for(j=JMIN;j<=JMAX;j++)
        {
            wy=dt/yy[j];
            for(kx=0;kx<Qx;kx++)
                for(ky=0;ky<Qy;ky++)
                { 	
                    grady=ey[ky]*(fy[j][kx][ky]-fy[j-1][kx][ky]);
                    f[j][i][kx][ky]=f[j][i][kx][ky]-wy*grady;		 

                    grady=ey[ky]*(hy[j][kx][ky]-hy[j-1][kx][ky]);
                    h[j][i][kx][ky]=h[j][i][kx][ky]-wy*grady;
                }
        }
    }
#pragma omp parallel for private(j,kx,ky)
    for(i=IMIN;i<=IMAX;i++)
        for(j=JMIN;j<=JMAX;j++)
        {
            rho[j][i]=0.0;u[j][i]=0.0;v[j][i]=0.0;te[j][i]=0.0;
            for(kx=0;kx<Qx;kx++)
                for(ky=0;ky<Qy;ky++)
                {
                    rho[j][i]+=f[j][i][kx][ky];
                    u[j][i]+=f[j][i][kx][ky]*ex[kx];
                    v[j][i]+=f[j][i][kx][ky]*ey[ky];
                    te[j][i]+=(ex[kx]*ex[kx]+ey[ky]*ey[ky])*f[j][i][kx][ky]+h[j][i][kx][ky];
                }

            u[j][i]=u[j][i]/rho[j][i];
            v[j][i]=v[j][i]/rho[j][i];
            te[j][i]=0.5*te[j][i]/rho[j][i];
            te[j][i]=(te[j][i]-0.5*(u[j][i]*u[j][i]+v[j][i]*v[j][i]))/Cv;
        }



}


void mesh()
{
    int i,j;
    double al,ar;
    double max=3.5;
    double may=5.5;
    FILE *fp;
    /*------generate mesh in X-direction---------------*/
    for(i=IMIN;i<=IMAX;i++)		
    {
        al=0.5+0.5*tanh(max*((i-1.)/NX-0.5))/tanh(0.5*max);
        ar=0.5+0.5*tanh(max*((i-0.)/NX-0.5))/tanh(0.5*max);
        x[i]=0.5*(al+ar)*Lx;
    }

    xx[IMIN]=2.*x[IMIN];
    for(i=IMIN+1;i<=IMAX;i++)
    {
        xx[i]=(x[i]-(x[i-1]+0.5*xx[i-1]))*2;
    }
    xx[IMIN-1]=xx[IMIN];xx[IMAX+1]=xx[IMAX];
    x[IMIN-1]=-0.5*xx[IMIN-1];
    x[IMAX+1]=x[IMAX]+0.5*(xx[IMAX]+xx[IMAX+1]);

    /*------generate mesh in Y-direction---------------*/
    for(j=JMIN;j<=JMAX;j++)		
    {
        al=0.5+0.5*tanh(may*((j-1.)/NY-0.5))/tanh(0.5*may);
        ar=0.5+0.5*tanh(may*((j-0.)/NY-0.5))/tanh(0.5*may);
        y[j]=0.5*(al+ar)*Ly;
    }

    yy[JMIN]=2.*y[JMIN];
    for(j=IMIN+1;j<=JMAX;j++)
    {
        yy[j]=(y[j]-(y[j-1]+0.5*yy[j-1]))*2;
    }
    yy[JMIN-1]=yy[JMIN];yy[JMAX+1]=yy[JMAX];
    y[JMIN-1]=-0.5*yy[JMIN-1];
    y[JMAX+1]=y[JMAX]+0.5*(yy[JMAX]+yy[JMAX+1]);
    /*--------set the minimum mesh -------------*/
    if(xx[IMIN]<yy[JMIN])
        xmin=xx[IMIN];
    else 
        xmin=yy[JMIN];
    /*--------save the mesh -------------*/
    if((fp=fopen("X.dat","w"))==NULL)
    {
        printf(" open file error\n");
        exit(1);
    }      
    for(i=IMIN-1;i<=IMAX+1;i++)
        fprintf(fp,"%.8e %.8e \n",x[i],xx[i]);  
    fclose(fp);	

    if((fp=fopen("Y.dat","w"))==NULL)
    {
        printf(" open file error\n");
        exit(1);
    }      
    for(j=JMIN-1;j<=JMAX+1;j++)
        fprintf(fp,"%.8e %.8e \n",y[j],yy[j]);  
    fclose(fp);	
}


#if VV==1
void setVelocity()
{
    int k;
    //half-range Gauss-Hermit quadrature
    double GH_v[14]=  {
        2.396724757843284e-002, 1.242911568927177e-001,  2.974477691473801e-001,  5.334998891822554e-001,   8.220981923307155e-001,
        1.154334829676328e+000, 1.523573280902749e+000,  1.925656879418240e+000,  2.358931078010347e+000,   2.824429077761555e+000,
        3.326604606373028e+000, 3.875436187413726e+000,  4.492761858309454e+000,  5.238744144080820e+000 };

    double GH_w[14]={
        6.123724869886580e-002, 1.361115667521515e-001,  1.889014205320117e-001,  1.985861257727940e-001,  1.585835386099182e-001,
        9.276908911579565e-002, 3.789962589474001e-002,  1.024419454744042e-002,  1.720120272586066e-003,  1.656430867082354e-004,
        8.177599975023724e-006, 1.734291294554730e-007,  1.139604105901161e-009,  1.037777563554779e-012 };

    cmax=GH_v[13];

    for(k=0;k<14;k++)
    {
        ex[k]=-GH_v[13-k]; ex[14+k]=GH_v[k];
        tpx[k]=GH_w[13-k]; tpx[14+k]=GH_w[k];
    }

    for(k=0;k<Qy;k++) {ey[k]=ex[k]; tpy[k]=tpx[k];}
    for(k=0;k<Qy;k++) {tpy[k]=tpx[k]=tpx[k]*exp(ex[k]*ex[k]);} // Requirement: 2*R*T_ref=1.0
}
#elif VV==2
void setVelocity()
{
    int k,hq;
    double power_law,Vmax;
    hq=Qx/2;
    Vmax=4.;

    double GH_v[hq], GH_w[hq];
    power_law=3;
    for(k=0;k<hq;k++)
    {
        GH_v[k]=Vmax*pow(k+0.5,power_law)/pow(hq+0.5,power_law);
        GH_w[k]=Vmax*power_law*pow(k+0.5,power_law-1)/pow(hq+0.5,power_law);

    }

    cmax=Vmax;

    for(k=0;k<hq;k++)
    {
        ex[k]=-GH_v[hq-1-k]; ex[hq+k]=GH_v[k];
        tpx[k]=GH_w[hq-1-k]; tpx[hq+k]=GH_w[k];
    }

    for(k=0;k<Qy;k++) {ey[k]=ex[k]; tpy[k]=tpx[k];}
}
#else
#endif

double feq(int kx,int ky,double RHO,double U,double V,double TT)
{
    double re,eu,ev;
    eu=ex[kx]-U;
    ev=ey[ky]-V;
    re=tpx[kx]*tpy[ky]*RHO*exp(-(eu*eu+ev*ev)/(2*R*TT))/(2*PI*R*TT);
    return(re);
}
void Shokhov(int kx,int ky, double RHO, double U, double V,double TT,double qx,double qy)
{
    double cx,cy,fpr,hpr,feqh,rh,vh,uh,th;
    rh=RHO;
    uh=U;
    vh=V;
    th=TT;

    cx=ex[kx]-uh;cy=ey[ky]-vh;
    fpr=(1-Pr)*(cx*qx+cy*qy)*((cx*cx+cy*cy)/(R*th)-D-2)/(5*rh*R*R*th*th);
    hpr=(1-Pr)*(cx*qx+cy*qy)*(((cx*cx+cy*cy)/(R*th)-D)*(K+3-D)-2*K)*R*th/(5*rh*R*R*th*th);
    feqh=feq(kx,ky,rh,uh,vh,th);
    fs=(1+fpr)*feqh;
    hs=((K+3-D)*R*th+hpr)*feqh;
}


#if sp==0
void slop()
{

    int j,i,kx,ky;

#pragma omp parallel for private(i,kx,ky)
    for(j=JMIN;j<=JMAX;j++)
        for(i=IMIN;i<=IMAX;i++)
            for(kx=0;kx<Qx;kx++)
                for(ky=0;ky<Qy;ky++)
                {
                    gfx[j][i][kx][ky]=(fb[j][i+1][kx][ky]-fb[j][i-1][kx][ky])/(x[i+1]-x[i-1]);
                    gfy[j][i][kx][ky]=(fb[j+1][i][kx][ky]-fb[j-1][i][kx][ky])/(y[j+1]-y[j-1]);

                    ghx[j][i][kx][ky]=(hb[j][i+1][kx][ky]-hb[j][i-1][kx][ky])/(x[i+1]-x[i-1]);
                    ghy[j][i][kx][ky]=(hb[j+1][i][kx][ky]-hb[j-1][i][kx][ky])/(y[j+1]-y[j-1]);
                }
    i=IMIN-1;
#pragma omp parallel for private(kx,ky)
    for(j=JMIN;j<=JMAX;j++)
        for(kx=0;kx<Qx;kx++)
            for(ky=0;ky<Qy;ky++)
            {
                gfx[j][i][kx][ky]=(fb[j][i+1][kx][ky]-fb[j][i][kx][ky])/(x[i+1]-x[i]);
                ghx[j][i][kx][ky]=(hb[j][i+1][kx][ky]-hb[j][i][kx][ky])/(x[i+1]-x[i]);

                gfy[j][i][kx][ky]=(fb[j+1][i][kx][ky]-fb[j-1][i][kx][ky])/(y[j+1]-y[j-1]);
                ghy[j][i][kx][ky]=(hb[j+1][i][kx][ky]-hb[j-1][i][kx][ky])/(y[j+1]-y[j-1]);
            }				       					
    i=IMAX+1;
#pragma omp parallel for private(kx,ky)
    for(j=JMIN;j<=JMAX;j++)
        for(kx=0;kx<Qx;kx++)
            for(ky=0;ky<Qy;ky++)
            {
                gfx[j][i][kx][ky]=(fb[j][i][kx][ky]-fb[j][i-1][kx][ky])/(x[i]-x[i-1]);
                ghx[j][i][kx][ky]=(hb[j][i][kx][ky]-hb[j][i-1][kx][ky])/(x[i]-x[i-1]);

                gfy[j][i][kx][ky]=(fb[j+1][i][kx][ky]-fb[j-1][i][kx][ky])/(y[j+1]-y[j-1]);
                ghy[j][i][kx][ky]=(hb[j+1][i][kx][ky]-hb[j-1][i][kx][ky])/(y[j+1]-y[j-1]);
            }				       					
    j=JMIN-1;
#pragma omp parallel for private(kx,ky)
    for(i=IMIN;i<=IMAX;i++)
        for(kx=0;kx<Qx;kx++)
            for(ky=0;ky<Qy;ky++)
            {

                gfy[j][i][kx][ky]=(fb[j+1][i][kx][ky]-fb[j][i][kx][ky])/(y[j+1]-y[j]);
                ghy[j][i][kx][ky]=(hb[j+1][i][kx][ky]-hb[j][i][kx][ky])/(y[j+1]-y[j]);

                gfx[j][i][kx][ky]=(fb[j][i+1][kx][ky]-fb[j][i-1][kx][ky])/(x[i+1]-x[i-1]);
                ghx[j][i][kx][ky]=(hb[j][i+1][kx][ky]-hb[j][i-1][kx][ky])/(x[i+1]-x[i-1]);
            }
    j=JMAX+1;
#pragma omp parallel for private(kx,ky)
    for(i=IMIN;i<=IMAX;i++)
        for(kx=0;kx<Qx;kx++)
            for(ky=0;ky<Qy;ky++)
            {

                gfy[j][i][kx][ky]=(fb[j][i][kx][ky]-fb[j-1][i][kx][ky])/(y[j]-y[j-1]);
                ghy[j][i][kx][ky]=(hb[j][i][kx][ky]-hb[j-1][i][kx][ky])/(y[j]-y[j-1]);

                gfx[j][i][kx][ky]=(fb[j][i+1][kx][ky]-fb[j][i-1][kx][ky])/(x[i+1]-x[i-1]);
                ghx[j][i][kx][ky]=(hb[j][i+1][kx][ky]-hb[j][i-1][kx][ky])/(x[i+1]-x[i-1]);
            }
}
#elif sp==1
void slop()
{

    int j,i,kx,ky;
    double s1,s2;

#pragma omp parallel for private(i,kx,ky,s1,s2)
    for(j=JMIN;j<=JMAX;j++)
        for(i=IMIN;i<=IMAX;i++)
            for(kx=0;kx<Qx;kx++)
                for(ky=0;ky<Qy;ky++)
                {
                    s1=(fb[j][i][kx][ky]-fb[j][i-1][kx][ky])/dx;
                    s2=(fb[j][i+1][kx][ky]-fb[j][i][kx][ky])/dx;
                    gfx[j][i][kx][ky]=(sgn(s1)+sgn(s2))*fabs(s1)*fabs(s2)/(fabs(s1)+fabs(s2)+1.0e-15);
                    s1=(fb[j][i][kx][ky]-fb[j-1][i][kx][ky])/dy;
                    s2=(fb[j+1][i][kx][ky]-fb[j][i][kx][ky])/dy;
                    gfy[j][i][kx][ky]=(sgn(s1)+sgn(s2))*fabs(s1)*fabs(s2)/(fabs(s1)+fabs(s2)+1.0e-15);

                    s1=(hb[j][i][kx][ky]-hb[j][i-1][kx][ky])/dx;
                    s2=(hb[j][i+1][kx][ky]-hb[j][i][kx][ky])/dx;
                    ghx[j][i][kx][ky]=(sgn(s1)+sgn(s2))*fabs(s1)*fabs(s2)/(fabs(s1)+fabs(s2)+1.0e-15);
                    s1=(hb[j][i][kx][ky]-hb[j-1][i][kx][ky])/dy;
                    s2=(hb[j+1][i][kx][ky]-hb[j][i][kx][ky])/dy;
                    ghy[j][i][kx][ky]=(sgn(s1)+sgn(s2))*fabs(s1)*fabs(s2)/(fabs(s1)+fabs(s2)+1.0e-15);
                }
    i=IMIN-1;
#pragma omp parallel for private(kx,ky,s1,s2)
    for(j=JMIN;j<=JMAX;j++)
        for(kx=0;kx<Qx;kx++)
            for(ky=0;ky<Qy;ky++)
            {
                gfx[j][i][kx][ky]=(fb[j][i+1][kx][ky]-fb[j][i][kx][ky])/dx;
                ghx[j][i][kx][ky]=(hb[j][i+1][kx][ky]-hb[j][i][kx][ky])/dx;

                s1=(fb[j][i][kx][ky]-fb[j-1][i][kx][ky])/dy;
                s2=(fb[j+1][i][kx][ky]-fb[j][i][kx][ky])/dy;
                gfy[j][i][kx][ky]=(sgn(s1)+sgn(s2))*fabs(s1)*fabs(s2)/(fabs(s1)+fabs(s2)+1.0e-15);

                s1=(hb[j][i][kx][ky]-hb[j-1][i][kx][ky])/dy;
                s2=(hb[j+1][i][kx][ky]-hb[j][i][kx][ky])/dy;
                ghy[j][i][kx][ky]=(sgn(s1)+sgn(s2))*fabs(s1)*fabs(s2)/(fabs(s1)+fabs(s2)+1.0e-15);
            }				       					
    i=IMAX+1;
#pragma omp parallel for private(kx,ky,s1,s2)
    for(j=JMIN;j<=JMAX;j++)
        for(kx=0;kx<Qx;kx++)
            for(ky=0;ky<Qy;ky++)
            {
                gfx[j][i][kx][ky]=(fb[j][i][kx][ky]-fb[j][i-1][kx][ky])/dx;
                ghx[j][i][kx][ky]=(hb[j][i][kx][ky]-hb[j][i-1][kx][ky])/dx;

                s1=(fb[j][i][kx][ky]-fb[j-1][i][kx][ky])/dy;
                s2=(fb[j+1][i][kx][ky]-fb[j][i][kx][ky])/dy;
                gfy[j][i][kx][ky]=(sgn(s1)+sgn(s2))*fabs(s1)*fabs(s2)/(fabs(s1)+fabs(s2)+1.0e-15);

                s1=(hb[j][i][kx][ky]-hb[j-1][i][kx][ky])/dy;
                s2=(hb[j+1][i][kx][ky]-hb[j][i][kx][ky])/dy;
                ghy[j][i][kx][ky]=(sgn(s1)+sgn(s2))*fabs(s1)*fabs(s2)/(fabs(s1)+fabs(s2)+1.0e-15);
            }				       					
    j=JMIN-1;
#pragma omp parallel for private(kx,ky,s1,s2)
    for(i=IMIN;i<=IMAX;i++)
        for(kx=0;kx<Qx;kx++)
            for(ky=0;ky<Qy;ky++)
            {

                gfy[j][i][kx][ky]=(fb[j+1][i][kx][ky]-fb[j][i][kx][ky])/dy;
                ghy[j][i][kx][ky]=(hb[j+1][i][kx][ky]-hb[j][i][kx][ky])/dy;

                s1=(fb[j][i][kx][ky]-fb[j][i-1][kx][ky])/dx;
                s2=(fb[j][i+1][kx][ky]-fb[j][i][kx][ky])/dx;
                gfx[j][i][kx][ky]=(sgn(s1)+sgn(s2))*fabs(s1)*fabs(s2)/(fabs(s1)+fabs(s2)+1.0e-15);

                s1=(hb[j][i][kx][ky]-hb[j][i-1][kx][ky])/dx;
                s2=(hb[j][i+1][kx][ky]-hb[j][i][kx][ky])/dx;
                ghx[j][i][kx][ky]=(sgn(s1)+sgn(s2))*fabs(s1)*fabs(s2)/(fabs(s1)+fabs(s2)+1.0e-15);
            }
    j=JMAX+1;
#pragma omp parallel for private(kx,ky,s1,s2)
    for(i=IMIN;i<=IMAX;i++)
        for(kx=0;kx<Qx;kx++)
            for(ky=0;ky<Qy;ky++)
            {

                gfy[j][i][kx][ky]=(fb[j][i][kx][ky]-fb[j-1][i][kx][ky])/dy;
                ghy[j][i][kx][ky]=(hb[j][i][kx][ky]-hb[j-1][i][kx][ky])/dy;

                s1=(fb[j][i][kx][ky]-fb[j][i-1][kx][ky])/dx;
                s2=(fb[j][i+1][kx][ky]-fb[j][i][kx][ky])/dx;
                gfx[j][i][kx][ky]=(sgn(s1)+sgn(s2))*fabs(s1)*fabs(s2)/(fabs(s1)+fabs(s2)+1.0e-15);

                s1=(hb[j][i][kx][ky]-hb[j][i-1][kx][ky])/dx;
                s2=(hb[j][i+1][kx][ky]-hb[j][i][kx][ky])/dx;
                ghx[j][i][kx][ky]=(sgn(s1)+sgn(s2))*fabs(s1)*fabs(s2)/(fabs(s1)+fabs(s2)+1.0e-15);
            }
}
#else
#endif
