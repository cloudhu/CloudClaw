function [data,rtn,retrace,order,indicators]=DemoStrategy_HighGrowth(startyear,endyear,sector,index,capital)
%%
%高成长择股策略
%组合权重：市值加权
%组合更新频率：年
%
%输入:
%	startyear=2012;   %起始年
%	endyear=2017; %截止年
%   sector='009007063';   %板块代码
%   index='000016.SH';  %作图比较的指数代码
%	captial=10^7;  %资金
%输出:
%   data：[日期,持仓明细,仓位变动明细,持仓金额,流动资金,净值,收益率]
%   date：日期
%   rtn：期间收益率
%   retrace：回撤
%	order：仓位变动明细
%   indicators：data的表头

%%
%登录API并获取数据
warning off
em=EmQuantMatlab();
errorid=em.start('forcelogin=1');
%输入默认值
if ~exist('startyear','var')
    startyear=2012;
    endyear=2017;
    sector='009007063';
    index='000016.SH';  %作图比较
    capital=10^7;  %资金
end
%提取股票代码，筛选startyear-3年前上市的股票
%提取交易月日期序列
startdate=strcat(num2str(startyear-1),'/12/31');%20111231
enddate=strcat(num2str(endyear),'/12/31');%20171231
codes0=em.sector(sector,'2017/10/10');
listdate=em.css(codes0(:,1),'LISTDATE');
reportdate1=datenum(strcat(num2str(startyear-4),'/01/01'),'yyyy/mm/dd');%20080101
codes0(find(datenum(char(listdate))>reportdate1),:)=[];
date=em.tradedates(strcat(num2str(startyear),'/04/01'),enddate,'Period=2');
%计算调仓次数，3月可获取到上一年年报信息
if endyear==str2double(datestr(today,'yyyy')) && str2double(datestr(today,'mm'))<4
    y=endyear-startyear+3; 
else
    y=endyear-startyear+4;  
end
%%
%提取年度财务数据：
%   通过css调取EPS、销售收入、净利润同比增长率、营业收入同比增长率、净资产同比增长率
%   计算3年EPS、销售收入增长均值
%   FinancialData=[3年EPS增长率均值,3年销售收入增长率均值,净利润同比增长率、
%                   营业收入同比增长率、净资产同比增长率]
%   进行条件筛选
%   codes{:,3}=符合条件的股票代码

codes=cell(y-3,3);
for n=1:y %2007-2016年度财务数据
    reportdate=strcat(num2str(startyear-5+n),'/12/31');
	varargin1=strcat('ReportDate=',reportdate);
    EpsMr(:,:,n)=import_css(em,codes0,'EPSBASIC,MBREVENUE',varargin1); %EPS、销售收入
    if n>3
        FinancialData(:,1,n-3)=(EpsMr(:,1,n)-EpsMr(:,1,n-1))./EpsMr(:,1,n-1)/3+...
            (EpsMr(:,1,n-1)-EpsMr(:,1,n-2))./EpsMr(:,1,n-2)/3+...
            (EpsMr(:,1,n-2)-EpsMr(:,1,n-3))./EpsMr(:,1,n-3)/3;  %3年EPS增长率均值
        FinancialData(:,2,n-3)=(EpsMr(:,2,n)-EpsMr(:,2,n-1))./EpsMr(:,2,n-1)/3+...
            (EpsMr(:,2,n-1)-EpsMr(:,2,n-2))./EpsMr(:,2,n-2)/3+...
            (EpsMr(:,2,n-2)-EpsMr(:,2,n-3))./EpsMr(:,2,n-3)/3;  %3年销售收入增长率均值
        %净利润同比增长率、营业收入同比增长率、净资产同比增长率
        FinancialData(:,3:5,n-3)=import_css(em,codes0,'YOYNI,YOYGR,YOYEQUITY',varargin1);
        incpsindicators2=strcat('s,PEG,',reportdate,',6');
        %条件筛选%%%%%%%%%%%%%%
        err1=1;  %断线重连判断参数
        while err1~=0
            try
                [codes{n-3,1},~,~,~,err1]=em.cps(codes0(:,1),incpsindicators2,'[s]<1');
            catch
                fprintf('重连中...')
                errorid=em.start('forcelogin=1');
            end
        end
        try
            if isnan(codes{n-3,1}),codes{n-3,1}=[];end %无符合条件股票则取[];
        end
        yr_codes0=codes0;yr_FinancialData=FinancialData(:,:,n-3);
        [m1,~]=find(isnan(yr_FinancialData));yr_FinancialData(m1,:)=[];yr_codes0(m1,:)=[];%去除无效数据
        [m2,~]=sort(yr_FinancialData(:,1),'descend');
        [m3,~]=sort(yr_FinancialData(:,2),'descend');
        [m4,~]=find(yr_FinancialData(:,1)>m2(fix(size(m2,1)/2))...
            & yr_FinancialData(:,2)>m3(fix(size(m3,1)/2)) & all(yr_FinancialData>0,2));
        codes{n-3,2}=yr_codes0(m4,1);
        codes{n-3,3}=intersect(codes{n-3,1},codes{n-3,2});    %符合条件的股票代码
    end
end

%%
%计算仓位及收益情况：
%   每年年末调仓
%   当年未筛选出合适股票：空仓
%   data=[日期,持仓明细,仓位变动明细,持仓金额,流动资金,净值,收益率]
%   data{:,2}=[持仓明细]=[代码,股票名称,收盘价,市值,持股比例,持股数,持仓金额]
%   data{:,3}=[仓位变动明细]=[代码,股票名称,日期,买卖类型,股数,交易价格]

indicators=[{'日期'},{'持仓明细'},{'仓位变动明细'},{'持仓金额'},{'流动资金'},{'净值'},{'收益率'}];
data(:,1)=cellstr(datestr(datenum(char(date())),'yyyy/mm/dd'));
buy=0;%判断是否初次持仓参数，初次持仓时变为1
chgMon=[str2num(cell2mat(data(:,1))),zeros(size(data,1),2)];
chgMon(:,2)=cell2mat(cellfun(@(x) str2num(x(6:7)),data(:,1),'UniformOutput',false));
order=[];
%计算日期对应财政年度;
for n=1:size(date,1)
    if chgMon(n,2)>3
        chgMon(n,3)=str2num(data{n,1}(1:4))-1;
    else
        chgMon(n,3)=str2num(data{n,1}(1:4))-2;
    end
end
for n=1:size(data,1)
    if n== 258
        disp(n)
    else
        disp(n)
    end
    varargin2=strcat('TradeDate=',data{n,1},',AdjustFlag=2');
    if n==1 %第一天
        oldportfolio=cell(0,4);oldcapital=capital;
    elseif isempty(data{n-1,2})  %如果未持仓
        oldportfolio=cell(0,4);oldcapital=data{n-1,5};
    else
        %oldportfolio为旧组合在调仓日的价格、持仓情况
        oldportfolio=[data{n-1,2}(:,1),import_css(em,data{n-1,2}(:,1),'NAME,Close',varargin2),...
            data{n-1,2}(:,6)];%旧组合在调仓日的[代码,名称,收盘价,持股数]
        oldcapital=sum(cell2mat(oldportfolio(:,3)).*cell2mat(oldportfolio(:,4)))+data{n-1,5};
    end
    %计算data{n,2}=[持仓明细]=[代码,股票名称,收盘价,市值,持股比例,持股数,持仓金额]
    
    data{n,2}(:,1)=codes{chgMon(n,3)-startyear+2,3};    %持仓明细
    data_2=zeros(size(data{n,2}(:,1),1),7);
    if isempty(codes{chgMon(n,3)-startyear+2,3})
        data{n,2}=cell(0,7);    %如果未选出股票
    else
        data{n,2}(:,2:4)=import_css(em,data{n,2}(:,1),'NAME,Close,MV',varargin2);
        data_2(:,3:4)=cell2mat(data{n,2}(:,3:4));
        if isequal(oldportfolio(:,1),data{n,2}(:,1))
            data_2(:,6)=cell2mat(data{n-1,2}(:,6));
        else
            data_2(:,5)=data_2(:,4)./sum(data_2(:,4)); %持股比例
            data_2(:,6)=fix(oldcapital*data_2(:,5)./data_2(:,3)/100)*100;    %股数
        end
        data_2(:,7)=data_2(:,3).*data_2(:,6);  %持仓金额
        data_2(:,5)=data_2(:,7)./sum(data_2(:,7)); %持股比例
        data{n,2}(:,5:7)=num2cell(data_2(:,5:7));
    end
    %计算仓位变动明细=[代码,股票名称,日期,买卖类型,股数,交易价格]
    [m3,c1,c2]=intersect(oldportfolio(:,1),data{n,2}(:,1));
    [~,c3]=setdiff(data{n,2}(:,1),m3);
    [~,c4]=setdiff(oldportfolio(:,1),m3);
    change=[];
    change=[m3,data{n,2}(c2,2),...
        num2cell(cell2mat(data{n,2}(c2,6))-cell2mat(oldportfolio(c1,4))),...
        data{n,2}(c2,3)];%  调仓股票
    change=[change;data{n,2}(c3,[1,2,6,3])];    %新买入股票
    change=[change;oldportfolio(c4,1:2),num2cell(-cell2mat(oldportfolio(c4,4))),...
        oldportfolio(c4,3)];    %清仓股票
    data{n,3}=[change(:,1:2),repmat([data{n,1},{'买入'}],size(change,1),1),change(:,3:4)];
    data{n,3}(find(cell2mat(data{n,3}(:,5))<0),4)={'卖出'};
    data{n,3}(find(cell2mat(data{n,3}(:,5))==0),:)=[];
    data{n,4}=sum(data_2(:,7));  %持仓金额
    if size(data{n,3},1)~=0 || n==1
        data{n,5}=oldcapital-data{n,4};   %流动资金,调仓日
    else
        data{n,5}=data{n-1,5};  %流动资金，非调仓日
    end
    data{n,6}=data{n,4}+data{n,5};  %净值
    data{n,7}=(data{n,6}-capital)/capital;  %收益率
    order=[order;data{n,3}];
end


%%
%求最大回撤率、总收益率
r2=0;
for i=1:n
	for j=i:n
        r1=(data{i,6}-data{j,6})/data{j,6};
        r2=[r2,r1];
	end
end
retrace=max(r2); %回撤率
rtn=data{end,7};    %总收益率

%%
%作图并保存数据
err2=1;
while err2~=0
    try
        [y1,~,~,~,err2]=em.csd(index,'CLOSE',date{1,1},date{end,1},'Period=2');
    catch
        fprintf('重连中...')
        errorid=em.start('forcelogin=1');
    end
end

y0=cell2mat(data(:,6));
y2=y1(1,1)/y0(1,1)*y0;
x1=datenum(datestr(char(date)));
xmin=x1(1);
xmax=x1(end);
ymin=min([y1;y2]);
ymax=max([y1;y2]);
close all
figure (1)
hold on

set(gcf,'unit','centimeters','position',[3 5 30 15])
plot(x1,y1,'Color',[91,155,213]/255,'linewidth',2)
plot(x1,y2,'Color',[238,84,84]/255,'linewidth',2)
axis([xmin xmax ymin ymax]);
set(gca,'XTick',x1(1:fix(size(x1,1))/10:end))
xname=cellstr(datestr(x1(1:fix(size(x1,1))/10:end),'yyyy/mm/dd'))';
set(gca,'XTickLabel',xname,'FontSize',6.5) 
legend(index,'Return','Location','EastOutside');
grid on
xlabel(strcat('Time:',num2str(startyear),'-',num2str(endyear),'   Return:',...
    num2str(rtn),'   Retrace:',num2str(retrace)),'FontSize',8)
saveas(figure(1),'Demo.jpg')
close all
save('Demo.mat','data','rtn','order','retrace')
em.close

end

function datas=import_css(em,incodes,inindicators,varargin)
erroid=1;%断线重连判断参数
while erroid~=0
    try
        [datas,~,~,~,erroid]=em.css(incodes,inindicators,cell2mat(varargin));
    catch
        %pause(5)
        fprintf('重连中...')
        errorid=em.start('forcelogin=1');
    end
end
end