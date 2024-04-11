//Copyright 2024 Andrey S. Ionisyan (anserion@gmail.com)
//
//Licensed under the Apache License, Version 2.0 (the "License");
//you may not use this file except in compliance with the License.
//You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
//Unless required by applicable law or agreed to in writing, software
//distributed under the License is distributed on an "AS IS" BASIS,
//WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//See the License for the specific language governing permissions and
//limitations under the License.

unit Unit1;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Forms, Controls, Graphics, Dialogs, StdCtrls, ExtCtrls,
  ExtDlgs, FPCanvas, LCLintf, LCLType;

type

  { TForm1 }

  TForm1 = class(TForm)
    Bevel1: TBevel;
    Bevel_receptors: TBevel;
    BTN_train_receptor: TButton;
    BTN_tune_art: TButton;
    BTN_out_gen_to_receptors: TButton;
    BTN_s_clear_white: TButton;
    BTN_s_clear_gray: TButton;
    BTN_s_noise: TButton;
    BTN_train: TButton;
    BTN_create_art: TButton;
    BTN_nw_reset_generator: TButton;
    BTN_s_clear_black: TButton;
    BTN_samples_load: TButton;
    BTN_s_save: TButton;
    BTN_s_random: TButton;
    BTN_s_load: TButton;
    BTN_out_gen_save: TButton;
    CB_contrast: TCheckBox;
    CB_autolevel: TCheckBox;
    Edit_cur_sample: TEdit;
    Edit_contrast: TEdit;
    Edit_N_neurons: TEdit;
    Label13: TLabel;
    Label3: TLabel;
    Label_Layer3: TLabel;
    Label2: TLabel;
    OpenPictureDialog: TOpenPictureDialog;
    PB_generator: TPaintBox;
    PB_receptors: TPaintBox;
    SB_samples: TScrollBar;
    Timer1: TTimer;
    procedure BTN_train_receptorClick(Sender: TObject);
    procedure BTN_tune_artClick(Sender: TObject);
    procedure BTN_out_gen_to_receptorsClick(Sender: TObject);
    procedure BTN_samples_loadClick(Sender: TObject);
    procedure BTN_out_gen_saveClick(Sender: TObject);
    procedure BTN_s_clear_whiteClick(Sender: TObject);
    procedure BTN_s_clear_grayClick(Sender: TObject);
    procedure BTN_s_noiseClick(Sender: TObject);
    procedure BTN_create_artClick(Sender: TObject);
    procedure BTN_nw_reset_generatorClick(Sender: TObject);
    procedure BTN_s_clear_blackClick(Sender: TObject);
    procedure BTN_s_loadClick(Sender: TObject);
    procedure BTN_s_randomClick(Sender: TObject);
    procedure BTN_s_saveClick(Sender: TObject);
    procedure BTN_trainClick(Sender: TObject);
    procedure CB_autolevelChange(Sender: TObject);
    procedure CB_contrastChange(Sender: TObject);
    procedure FormCreate(Sender: TObject);
    procedure PB_generatorPaint(Sender: TObject);
    procedure PB_receptorsPaint(Sender: TObject);
    procedure SB_samplesChange(Sender: TObject);
    procedure Timer1Timer(Sender: TObject);
  private

  public

  end;

const
  s_width=64;
  s_height=64;

type
  TIntegerVector = array of integer;
  TRealVector = array of real;
  TRealMatrix = array of TRealVector;

var
  Form1: TForm1;
  receptorsBitmap,hopfieldBitmap: TBitmap;
  img_buffer:array[0..1023,0..1023]of real;
  S_elements,Z_elements:TRealMatrix;
  samples:array of TRealMatrix;
  sample:TRealVector;
  curSample:integer;

  n_hopfield_neurons:integer;
  hopfield_w:TRealMatrix;
  hopfield_input:TRealVector;
  hopfield_output:TRealVector;

implementation

{$R *.lfm}

function sigmoid(x:real):real;
begin sigmoid:=1/(1+exp(-x)); end;

function der_sigmoid(y:real):real;
begin der_sigmoid:=y*(1-y); end;

function tanh(x:real):real;
var y:real;
begin tanh:=(exp(x)-exp(-x))/(exp(x)+exp(-x)); end;

function der_tanh(y:real):real;
begin der_tanh:=1-y*y; end;

function ReLU(x:real):real;
begin if x<0 then ReLU:=0.01*x else ReLU:=x; end;

function der_ReLU(y:real):real;
begin if y<=0 then der_ReLU:=0.01 else der_ReLU:=1; end;

function activation(x:real):real;
begin activation:=2.0*(sigmoid(x)-0.5); end; //tanh(x); end;

function der_activation(y:real):real;
begin der_activation:=der_tanh(y); end;

//==============================================================
function scalarProduct(v1,v2:TRealVector):real;
var i,n:integer; res:real;
begin
  n:=length(v1);
  res:=0;
  for i:=0 to n-1 do res:=res+v1[i]*v2[i];
  scalarProduct:=res;
end;

procedure shuffleIntegerVector(vector:TIntegerVector);
var n,k,k1,k2:integer; tmp:integer;
begin
  n:=length(vector);
  //for k:=0 to n-1 do vector[k]:=k mod n;
  for k:=1 to n do
  begin
    k1:=random(n); k2:=random(n);
    tmp:=vector[k1]; vector[k1]:=vector[k2]; vector[k2]:=tmp;
  end;
end;

procedure valueToVector(vector:TRealVector; value:real);
var i,n:integer;
begin
  n:=length(vector);
  for i:=0 to n-1 do vector[i]:=value;
end;

procedure valueToMatrix(matrix:TRealMatrix; value:real);
var i,n:integer;
begin
  n:=length(matrix);
  for i:=0 to n-1 do valueToVector(matrix[i],value);
end;

procedure randomToVector(vector:TRealVector; min_value,max_value:real);
var i,n:integer;
begin
  n:=length(vector);
  for i:=0 to n-1 do vector[i]:=min_value+(max_value-min_value)*random;
end;

procedure randomToMatrix(matrix:TRealMatrix; min_value,max_value:real);
var i,n:integer;
begin
  n:=length(matrix);
  for i:=0 to n-1 do randomToVector(matrix[i],min_value,max_value);
end;

procedure shakeVector(vector:TRealVector; value:real);
var i,n:integer;
begin
  n:=length(vector);
  for i:=0 to n-1 do vector[i]:=vector[i]+2*value*(random-0.5);
end;

procedure shakeMatrix(matrix:TRealMatrix; value:real);
var i,n:integer;
begin
  n:=length(matrix);
  for i:=0 to n-1 do shakeVector(matrix[i],value);
end;

procedure noiseToVector(vector:TRealVector; noise_value:real);
var i,n:integer;
begin
  n:=length(vector);
  for i:=0 to n-1 do
    if random<=noise_value then vector[i]:=2.0*random-1.0;
end;

procedure noiseToMatrix(matrix:TRealMatrix; noise_value:real);
var i,n:integer;
begin
  n:=length(matrix);
  for i:=0 to n-1 do noiseToVector(matrix[i],noise_value);
end;

procedure MatrixToVector(matrix:TRealMatrix; vector:TRealVector);
var i,j,n,m,cnt:integer;
begin
  n:=length(matrix); m:=length(matrix[0]); cnt:=0;
  for i:=0 to n-1 do
  for j:=0 to m-1 do
  begin
    vector[cnt]:=matrix[i,j];
    cnt:=cnt+1;
  end;
end;

procedure VectorToMatrix(vector:TRealVector; matrix:TRealMatrix);
var i,j,n,m,cnt:integer;
begin
  n:=length(matrix); m:=length(matrix[0]); cnt:=0;
  for i:=0 to n-1 do
  for j:=0 to m-1 do
  begin
    matrix[i,j]:=vector[cnt];
    cnt:=cnt+1;
  end;
end;

procedure VectorCpy(src,dst:TRealVector);
var i,n:integer;
begin
     n:=length(src);
     for i:=0 to n-1 do dst[i]:=src[i];
end;

procedure MatrixCpy(src,dst:TRealMatrix);
var i,n:integer;
begin
     n:=length(src);
     for i:=0 to n-1 do VectorCpy(src[i],dst[i]);
end;

function VectorMin(vector:TRealVector):real;
var i,n:integer; res:real;
begin
     n:=length(vector);
     res:=vector[0];
     for i:=0 to n-1 do
         if vector[i]<res then res:=vector[i];
     VectorMin:=res;
end;

function VectorMax(vector:TRealVector):real;
var i,n:integer; res:real;
begin
     n:=length(vector);
     res:=vector[0];
     for i:=0 to n-1 do
         if vector[i]>res then res:=vector[i];
     VectorMax:=res;
end;

function MatrixMin(matrix:TRealMatrix):real;
var i,n:integer; tmp,res:real;
begin
     n:=length(matrix);
     res:=matrix[0,0];
     for i:=0 to n-1 do
     begin
       tmp:=VectorMin(matrix[i]);
       if tmp<res then res:=tmp;
     end;
     MatrixMin:=res;
end;

function MatrixMax(matrix:TRealMatrix):real;
var i,n:integer; tmp,res:real;
begin
     n:=length(matrix);
     res:=matrix[0,0];
     for i:=0 to n-1 do
     begin
       tmp:=VectorMax(matrix[i]);
       if tmp>res then res:=tmp;
     end;
     MatrixMax:=res;
end;

procedure MatrixToBitmap(matrix:TRealMatrix; bitmap:TBitmap; contrast:real; autoflag:boolean);
var x,y,sx,sy:integer; dx,dy:real; C_min,C_max,deltaC,C:real;
    dst_bpp:integer; dst_ptr:PByte; R,G,B:byte;
    n,m:integer;
begin
  n:=length(matrix);
  m:=length(matrix[0]);
  dx:=bitmap.Width/n;
  dy:=bitmap.Height/m;

  C_min:=MatrixMin(matrix); C_max:=MatrixMax(matrix);
  deltaC:=C_max-C_min;
  if deltaC=0 then autoflag:=false;
  for x:=0 to n-1 do
  for y:=0 to m-1 do
  begin
    if autoflag
    then C:=((matrix[x,y]-C_min)/deltaC-0.5)*2.0
    else C:=matrix[x,y];

    //C:=(C-0.5)*contrast+0.5; //sigmoid
    C:=C*contrast; //tahh
    //if C<0 then C:=0; //sigmoid
    if C<-1 then C:=-1; //tanh
    if C>1 then C:=1;
    C:=(C+1.0)*0.5; //tanh
    for sx:=0 to trunc(dx) do
    for sy:=0 to trunc(dy) do
      img_buffer[trunc(sx+x*dx),trunc(sy+y*dy)]:=C;
  end;

  bitmap.BeginUpdate(false);
  dst_ptr:=bitmap.RawImage.Data;
  dst_bpp:=bitmap.RawImage.Description.BitsPerPixel div 8;
  for y:=0 to bitmap.height-1 do
  for x:=0 to bitmap.width-1 do
  begin
     R:=trunc(img_buffer[x,y]*255); G:=R; B:=R;
     dst_ptr^:=B; (dst_ptr+1)^:=G; (dst_ptr+2)^:=R; inc(dst_ptr,dst_bpp);
  end;
  bitmap.EndUpdate(false);
end;

//=========================================================================

procedure nw_hopfield_allocation;
begin
  SetLength(hopfield_w,n_hopfield_neurons,n_hopfield_neurons);
  SetLength(hopfield_input,n_hopfield_neurons);
  SetLength(hopfield_output,n_hopfield_neurons);
  SetLength(sample,n_hopfield_neurons);
end;

procedure nw_hopfield_reset;
begin
  valueToVector(hopfield_input,0);
  valueToVector(hopfield_output,0);
  valueToMatrix(hopfield_w,0);
end;

procedure nw_hopfield_step;
var i,active_neuron:integer; tmp:real;
begin
  for i:=1 to 10*n_hopfield_neurons do
  begin
    active_neuron:=random(n_hopfield_neurons);
    tmp:=scalarProduct(hopfield_input,hopfield_w[active_neuron]);
    hopfield_output[active_neuron]:=activation(tmp);
    //hopfield_input[active_neuron]:=hopfield_output[active_neuron];
  end;
  for i:=0 to n_hopfield_neurons-1 do hopfield_input[i]:=hopfield_output[i];
end;

procedure nw_hopfield_train(sample:TRealVector);
var i,j:integer;
begin
  for i:=0 to n_hopfield_neurons-1 do
    for j:=0 to n_hopfield_neurons-1 do
      hopfield_w[i,j]:=hopfield_w[i,j]+sample[i]*sample[j];
  for i:=0 to n_hopfield_neurons-1 do hopfield_w[i,i]:=0;
end;

{ TForm1 }

procedure TForm1.PB_receptorsPaint(Sender: TObject);
begin
  MatrixToBitmap(S_elements,receptorsBitmap,1,CB_autolevel.Checked);
  PB_receptors.Canvas.Draw(0,0,receptorsBitmap);
end;

procedure TForm1.SB_samplesChange(Sender: TObject);
begin
  MatrixCpy(samples[SB_samples.Position],S_elements);
  PB_receptorsPaint(PB_receptors);
  Edit_cur_sample.text:=IntToStr(SB_samples.Position+1);
end;

procedure TForm1.Timer1Timer(Sender: TObject);
var i:integer;
begin
  if curSample<length(samples) then
  begin
    timer1.Enabled:=False;
    MatrixToVector(samples[curSample],sample);
    for i:=0 to n_hopfield_neurons-1 do sample[i]:=sample[i]/n_hopfield_neurons;
    nw_hopfield_train(sample);
    SB_samples.Position:=curSample;
    SB_samplesChange(self);
    BTN_create_artClick(self);
    curSample:=curSample+1;
    timer1.Enabled:=True;
  end
  else
  begin
    curSample:=0;
    BTN_trainClick(self);
  end;
end;

procedure TForm1.PB_generatorPaint(Sender: TObject);
var contrast_value:real;
begin
  if CB_contrast.Checked
  then contrast_value:=StrToFloat(Edit_contrast.text)/100
  else contrast_value:=1;
  MatrixToBitmap(Z_elements,hopfieldBitmap,contrast_value,CB_autolevel.Checked);
  PB_generator.Canvas.Draw(0,0,hopfieldBitmap);
end;

procedure TForm1.FormCreate(Sender: TObject);
begin
  randomize;

  SetLength(samples,0);
  SB_samples.max:=length(samples);

  receptorsBitmap:=TBitmap.Create;
  receptorsBitmap.SetSize(PB_receptors.width,PB_receptors.height);
  hopfieldBitmap:=TBitmap.Create;
  hopfieldBitmap.SetSize(PB_generator.width,PB_generator.height);

  SetLength(S_elements,s_width,s_height);
  SetLength(Z_elements,s_width,s_height);

  n_hopfield_neurons:=s_width*s_height;
  Edit_N_neurons.text:=IntToStr(n_hopfield_neurons);

  nw_hopfield_allocation;
  BTN_nw_reset_generatorClick(self);
  curSample:=0;
end;

procedure TForm1.BTN_nw_reset_generatorClick(Sender: TObject);
begin
  nw_hopfield_reset;
  MatrixToVector(S_elements,hopfield_input);
  nw_hopfield_step;
  VectorToMatrix(hopfield_output,Z_elements);
  PB_generatorPaint(PB_generator);
end;

procedure TForm1.BTN_create_artClick(Sender: TObject);
begin
  MatrixToVector(S_elements,hopfield_input);
  nw_hopfield_step;
  VectorToMatrix(hopfield_output,Z_elements);
  PB_generatorPaint(PB_generator);
end;

procedure TForm1.BTN_trainClick(Sender: TObject);
begin
  timer1.Enabled:=not(timer1.Enabled);
  if timer1.Enabled
  then BTN_train.Caption:='Стоп обучение'
  else BTN_train.Caption:='Обучить нейросеть';
end;

procedure TForm1.BTN_samples_loadClick(Sender: TObject);
var VideofileName: String;
    picture:TPicture;
    cell_x,cell_y,x,y,src_bpp:integer;
    src_ptr:PByte;
    R,G,B:word;
    dx,dy:real;
    C:real;
    k,n_samples:integer;
begin
  if OpenPictureDialog.execute then
  begin
    n_samples:=OpenPictureDialog.Files.Count;
    SetLength(samples,n_samples,s_width,s_height);
    picture:=TPicture.Create;
    for k:=0 to n_samples-1 do
    begin
      VideofileName:=OpenPictureDialog.Files[k];
      picture.LoadFromFile(VideofileName);

      src_ptr:=picture.Bitmap.RawImage.Data;
      src_bpp:=picture.Bitmap.RawImage.Description.BitsPerPixel div 8;
      for y:=0 to picture.Bitmap.height-1 do
      for x:=0 to picture.Bitmap.width-1 do
      begin
        R:=(src_ptr+2)^; G:=(src_ptr+1)^; B:=src_ptr^; inc(src_ptr,src_bpp);
        img_buffer[x,y]:=(R+G+B)/(256.0*3.0);
      end;

      dx:=picture.Bitmap.Width/s_width;
      dy:=picture.Bitmap.Height/s_height;
      for cell_x:=0 to s_width-1 do
      for cell_y:=0 to s_height-1 do
      begin
        C:=0;
        for x:=0 to trunc(dx) do
        for y:=0 to trunc(dy) do
          C:=C+img_buffer[trunc(x+cell_x*dx),trunc(y+cell_y*dy)];
        samples[k,cell_x,cell_y]:=C/((trunc(dx)+1)*(trunc(dy)+1)); //sigmoid
        samples[k,cell_x,cell_y]:=2.0*(samples[k,cell_x,cell_y]-0.5); //tahh
      end;
    end;
    picture.Free;
    curSample:=0;
    SB_samples.Position:=curSample;
    SB_samples.max:=length(samples)-1;
    SB_samplesChange(self);
  end;
end;

procedure TForm1.BTN_out_gen_to_receptorsClick(Sender: TObject);
begin
  MatrixCpy(Z_elements,S_elements);
  PB_receptorsPaint(PB_receptors);
end;

procedure TForm1.BTN_tune_artClick(Sender: TObject);
begin
  nw_hopfield_step;
  VectorToMatrix(hopfield_output,Z_elements);
  PB_generatorPaint(PB_generator);
end;

procedure TForm1.BTN_train_receptorClick(Sender: TObject);
var sample:TRealVector; i:integer;
begin
  SetLength(sample,n_hopfield_neurons);
  MatrixToVector(S_elements,sample);
  for i:=0 to n_hopfield_neurons-1 do sample[i]:=sample[i]/n_hopfield_neurons;
  nw_hopfield_train(sample);
  BTN_create_artClick(self);
end;

procedure TForm1.BTN_out_gen_saveClick(Sender: TObject);
begin
  hopfieldBitmap.SaveToFile('art_out.bmp');
end;

procedure TForm1.BTN_s_clear_whiteClick(Sender: TObject);
begin
  valueToMatrix(S_elements,1);
  PB_receptorsPaint(PB_receptors);
end;

procedure TForm1.BTN_s_clear_grayClick(Sender: TObject);
begin
  //valueToMatrix(S_elements,0.5); //sigmoid
  valueToMatrix(S_elements,0.0); //tanh
  PB_receptorsPaint(PB_receptors);
end;

procedure TForm1.BTN_s_noiseClick(Sender: TObject);
begin
  noiseToMatrix(S_elements,0.05);
  PB_receptorsPaint(PB_receptors);
end;

procedure TForm1.BTN_s_clear_blackClick(Sender: TObject);
begin
  //valueToMatrix(S_elements,0); //sigmoid
  valueToMatrix(S_elements,-1); //tanh
  PB_receptorsPaint(PB_receptors);
end;

procedure TForm1.BTN_s_loadClick(Sender: TObject);
var VideofileName: String;
    picture:TPicture;
    cell_x,cell_y,x,y,src_bpp:integer;
    src_ptr:PByte;
    R,G,B:word;
    dx,dy:real;
    C:real;
begin
  if OpenPictureDialog.execute then
  begin
      picture:=TPicture.Create;
      VideofileName:=OpenPictureDialog.FileName;
      picture.LoadFromFile(VideofileName);

      src_ptr:=picture.Bitmap.RawImage.Data;
      src_bpp:=picture.Bitmap.RawImage.Description.BitsPerPixel div 8;
      for y:=0 to picture.Bitmap.height-1 do
      for x:=0 to picture.Bitmap.width-1 do
      begin
          R:=(src_ptr+2)^; G:=(src_ptr+1)^; B:=src_ptr^; inc(src_ptr,src_bpp);
          img_buffer[x,y]:=(R+G+B)/(256.0*3.0);
      end;

      dx:=picture.Bitmap.Width/s_width;
      dy:=picture.Bitmap.Height/s_height;
      for cell_x:=0 to s_width-1 do
      for cell_y:=0 to s_height-1 do
      begin
          C:=0;
          for x:=0 to trunc(dx) do
          for y:=0 to trunc(dy) do
            C:=C+img_buffer[trunc(x+cell_x*dx),trunc(y+cell_y*dy)];
          S_elements[cell_x,cell_y]:=C/((trunc(dx)+1)*(trunc(dy)+1)); //sigmoid
          S_elements[cell_x,cell_y]:=2.0*(S_elements[cell_x,cell_y]-0.5); //tanh
      end;
      PB_receptorsPaint(PB_receptors);
  end;
end;

procedure TForm1.BTN_s_randomClick(Sender: TObject);
begin
  //randomToMatrix(S_elements,0,1); //sigmoid
  randomToMatrix(S_elements,-1,1); //tanh
  PB_receptorsPaint(PB_receptors);
end;

procedure TForm1.BTN_s_saveClick(Sender: TObject);
begin
  receptorsBitmap.SaveToFile('art_receptors.bmp');
end;

procedure TForm1.CB_contrastChange(Sender: TObject);
begin
  Edit_contrast.ReadOnly:=CB_contrast.Checked;
  PB_generatorPaint(PB_generator);
end;

procedure TForm1.CB_autolevelChange(Sender: TObject);
begin
  PB_receptorsPaint(PB_receptors);
  PB_generatorPaint(PB_generator);
end;

end.

