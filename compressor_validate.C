const int N = 6;

void compressor_validate(string folder = "./")
{  
  TMultiGraph** mg = new TMultiGraph*[N];
  for (int i = 0; i < N; i++) mg[i] = new TMultiGraph();

  TLegend *l = new TLegend(0.5,0.67,0.88,0.88);
  l->SetBorderSize(0);
  l->SetFillColor(0);
  l->SetFillStyle(0);

  int rep;
  double cv, md, dn50, up50, dn68, up68, dn90, up90, x;
  double mt1, mt2, mt3, mt4, mt5, mt6, mt7;

  fstream f,g;
  string file = folder + "erf_randomized.dat";
  f.open(file.c_str(), ios::in);

  TGraphErrors **gcv = new TGraphErrors*[N];
  TGraphErrors **gmd = new TGraphErrors*[N];
  TGraphErrors **g50 = new TGraphErrors*[N];
  TGraphErrors **g68 = new TGraphErrors*[N];
  TGraphErrors **g90 = new TGraphErrors*[N];
  for (int i = 0; i < N; i++) 
    {
      gcv[i] = new TGraphErrors();
      gcv[i]->SetMarkerStyle(20);
      gcv[i]->SetMarkerColor(kBlue);

      gmd[i] = new TGraphErrors();
      gmd[i]->SetMarkerStyle(24);
      gmd[i]->SetMarkerColor(kBlue);      

      g50[i] = new TGraphErrors();
      g50[i]->SetLineColor(kBlue); 

      g68[i] = new TGraphErrors();
      g68[i]->SetLineColor(kGreen+2); 

      g90[i] = new TGraphErrors();
      g90[i]->SetLineColor(kOrange); 
    }
  
  int index = 0;
  for (;;)
    {
      f >> x;    
      if (f.eof()) break;
      for (int i = 0; i < N; i++)
	{
	  f >> cv >> md >> dn50 >> up50 >> dn68 >> up68 >> dn90 >> up90;
	  gcv[i]->SetPoint(index, x, cv);
	  gmd[i]->SetPoint(index, x, md);

	  g50[i]->SetPoint(index, x, (up50+dn50)/2.0);
	  g50[i]->SetPointError(index, 0, up50-(up50+dn50)/2.0);

	  g68[i]->SetPoint(index, x, (up68+dn68)/2.0);
	  g68[i]->SetPointError(index, 0, up68-(up68+dn68)/2.0);

	  g90[i]->SetPoint(index, x, (up90+dn90)/2.0);
	  g90[i]->SetPointError(index, 0, up90-(up90+dn90)/2.0);
	}
      index++;
    }

  for (int i = 0; i < N; i++)
    {
      mg[i]->Add(g90[i],"p");
      mg[i]->Add(g68[i],"p");
      mg[i]->Add(g50[i],"p");
      mg[i]->Add(gcv[i],"p");
      mg[i]->Add(gmd[i],"p");
    }
  
  f.close();

  file = folder + "erf_reduced.dat";
  g.open(file.c_str(), ios::in); 
  
  TGraphErrors **gccv = new TGraphErrors*[N];
  for (int i = 0; i < N; i++) 
    {
      gccv[i] = new TGraphErrors();
      gccv[i]->SetMarkerStyle(20);
      gccv[i]->SetMarkerColor(kRed);
    }
  
  index = 0;
  for(;;)
    {
      g >> x;
      if (g.eof()) break;
      for (int i = 0; i < N; i++)
	{
	  g >> cv;
	  gccv[i]->SetPoint(index, x, cv);
	}
      index++;
    }

  for (int i = 0; i < N; i++)
    {
      mg[i]->Add(gccv[i],"p");
      
      if (i == 0)
	{
	  l->AddEntry(gccv[i],"Compressed","p");
	  l->AddEntry(gcv[i],"Random Mean (1k)","p");
	  l->AddEntry(gmd[i],"Random Median (1k)","p");
	  l->AddEntry(g50[i],"Random 50% c.l. (1k)","l");
	  l->AddEntry(g68[i],"Random 68% c.l. (1k)","l");
	  l->AddEntry(g90[i],"Random 90% c.l. (1k)","l");      
	}
    }
  g.close();

  string title[N];
  title[0] = "ERF Central Value";
  title[1] = "ERF Standard deviation";
  title[2] = "ERF Skewness";
  title[3] = "ERF Kurtosis";  
  title[4] = "ERF Kolmogorov";
  title[5] = "ERF Correlation";

  TCanvas *c = new TCanvas("c","Error function plots",1200,700);
  c->Divide(3,2);
  for (int i = 0; i < N; i++)
    {
      c->cd(i+1)->SetLogx();
      c->cd(i+1)->SetLogy();
      c->cd(i+1)->SetTickx();
      c->cd(i+1)->SetTicky();
      
      mg[i]->Draw("AP");
      mg[i]->SetTitle(title[i].c_str());
      mg[i]->GetXaxis()->SetTitle("Replicas");
      mg[i]->GetXaxis()->CenterTitle(true);
      l->Draw("same");
    }
}
