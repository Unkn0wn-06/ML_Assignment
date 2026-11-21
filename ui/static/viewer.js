const $=id=>document.getElementById(id);
const data=JSON.parse($('payload').textContent), frames=data.frames;
const palette=data.temperatureRamp;
function temperature(t){return palette[t>data.limit?6:Math.max(0,Math.min(5,Math.floor(t-data.limit+6)))];}
function format(v,d=2){return Number.isFinite(v)?v.toFixed(d):'—';}
function escapeHtml(v){return String(v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function safetyReading(t){return t>data.limit?{icon:'!',label:'Limit exceeded',color:'#f87171'}:t>=data.warning?{icon:'!',label:'Warning band',color:'#fbbf24'}:{icon:'✓',label:'Within limit',color:'#34d399'};}
let mode='3D',zone=null,pinned=null,timer=null,room;
const row=()=>frames[+$('step').value];
function rackName(index){return "Rack "+String.fromCharCode(65+Math.floor(index/10))+String(index%10+1).padStart(2,"0");}
function selectZone(index,lock){if(lock)pinned=pinned===index?null:index;zone=index??pinned;update();}
try{room=buildRoom($('room'),data,selectZone);}catch(error){$('room').innerHTML='<p class="fallback"><b>3D view unavailable</b><br>WebGL is unavailable. Use Loop or inspect the replay readings.</p>';}
const loop=makeLoop($('loop'),data);
const field=(name,val)=>'<div class="row"><span>'+name+'</span><b>'+val+'</b></div>';
const tile=(name,val,color='')=>'<div class="tile"><b style="color:'+color+'">'+val+'</b><small>'+name+'</small></div>';
const progress=(name,value,max,label)=>'<div class="group">'+field(name,label)+'<progress aria-label="'+name+'" value="'+Math.max(0,value)+'" max="'+max+'"></progress></div>';
const detail=(name,content)=>'<details><summary>'+name+'</summary><p>'+content+'</p></details>';
const spark=(key,color)=>{const values=frames.map(r=>r[key]),min=Math.min(...values),max=Math.max(...values);
  return '<svg class="spark" viewBox="0 0 100 32" preserveAspectRatio="none"><polyline fill="none" stroke="'+color+'" stroke-width="1.4" points="'+values.map((v,i)=>[i*100/Math.max(1,values.length-1),30-27*(v-min)/(max-min||1)].join(',')).join(' ')+'"/></svg>';};
const sparks=[spark('CoP','#fbbf24'),spark('P_DC','#5b9bff')];
function configList(){
  const lo=Math.min(...data.configs.map(c=>c.low)),hi=Math.max(...data.configs.map(c=>c.high)),pos=v=>100*(v-lo)/(hi-lo||1);
  return ['Baseline','Plate','PlateWall','Wall'].map(name=>{const c=data.configs.find(item=>item.name===name);
    if(!c)return field(name,'No evaluated episodes');return '<div class="group"><b>'+name+'</b> · PUE '+format(c.pue)+'<div class="range"><i style="left:'+pos(c.low)+'%;width:'+(pos(c.high)-pos(c.low))+'%"></i><b style="left:'+pos(c.mid)+'%"></b></div><small>'+format(c.low,1)+'–'+format(c.high,1)+' °C · median '+format(c.mid,1)+' °C</small><p style="color:'+(c.violated?'#fbbf24':'#34d399')+'">'+(c.violated?'⚠':'✓')+' '+c.violated+' / '+c.n+' episodes violated</p></div>';}).join('');
}
const configurations=configList(),limit=data.limit;
function setLegend(){
  const band=(title,labels,colors)=>'<section><h4>'+title+'</h4>'+labels.map((l,i)=>'<div class="tick"><i class="sw" style="background:'+colors[i]+'"></i>'+l+'</div>').join('')+'</section>';
  $('keys').innerHTML=mode==='LOOP'?band('Circuit',['Supply · cold','Return · hot'],['#5b9bff','#ef4a5c']):
    band('Outlet temp (°C)',['&lt; '+(limit-5),...Array.from({length:5},(_,i)=>(limit-5+i)+'–'+(limit-4+i)),'&gt; '+limit],palette)+
    band('Margin (°C)',['&gt; 4','2–4','1–2','0.5–1','0–0.5','&lt; 0'],['#2f6fe4','#4ade80','#a3e635','#facc15','#fb923c','#ef4444']);
}
function update(){
 const r=row(), s=safetyReading(r.outlet_temperature),margin=limit-r.outlet_temperature;
 const compare=$('compare').checked,peer=compare?data.comparison.find(o=>o.step===r.step):null;
 const decision=escapeHtml(r.decision_reason),cfg=escapeHtml(data.config),name=escapeHtml(data.controller);
 $('meta').innerHTML='<b>AAU · '+cfg+' · '+format(data.load,0)+'% load</b><small>'+name+' · seed '+data.seed+' · step '+r.step+'</small>';
 const oldOpen=new Set([...$('info').querySelectorAll('details[open] summary')].map(e=>e.textContent));
 let content='';
 if(mode==='LOOP'){
   $('title').textContent='Cooling loop';loop(r,temperature);
   const delta=Number.isFinite(data.baseline[r.step])?format(r.CoP-data.baseline[r.step]):'—';
   content='<div class="tiles">'+tile('SYSTEM CoP',format(r.CoP)+sparks[0])+tile('P_DC · kW',format(r.P_DC)+sparks[1])+'</div>'+field('CoP Δ vs rule-based',delta)+field('PUE',format(r.PUE))+'<div class="group"><h4>Configurations</h4></div>'+configurations;
 }else{
   $('title').textContent=zone===null?'Room inspector':rackName(zone)+' · schematic';
   room?.update(r,temperature,s);
   const cooling='Unavailable · cooling-only power is unmetered';
   content=field('Scenario',escapeHtml(data.scenario))+field('Step',r.step+' / '+data.horizon)+field('Controller',name)+field('Status','<span style="color:'+s.color+'">'+s.icon+' '+s.label+'</span>')+
   '<div class="tiles">'+tile('Outlet °C','<i class="sw" style="background:'+temperature(r.outlet_temperature)+'"></i>'+format(r.outlet_temperature,1))+tile('Supply °C',format(r.proposed_supply,1),'#76bbff')+tile('P_DC kW',format(r.P_DC))+'</div>'+
   progress('Margin to '+limit+' °C',margin,Math.max(5,margin),format(margin,2)+' °C')+progress('PUE',r.PUE,Math.max(...frames.map(f=>f.PUE)),format(r.PUE))+
   field('Decision',decision)+'<div class="group"><button id="toLoop">View cooling loop</button></div>'+
   detail('Readings','Inlet '+format(r['Inlet Temp Avg'])+' °C · return '+format(r['Cooling Return Average'])+' °C')+
   detail('Power','IT '+format(r.P_IT)+' kW · facility '+format(r.P_DC)+' kW. Cooling: '+cooling)+
   detail('Policy & decisions',decision+' · lower '+data.lower+' °C · warning '+data.warning+' °C')+
   detail('Alerts',s.icon+' '+s.label)+detail('Episode summary',escapeHtml(data.reason)+' · '+(frames.length-1)+' / '+data.horizon+' steps · '+data.switches+' switches · '+data.clipped+' clipped steps');
   $('zone').hidden=zone===null;
   if(zone!==null)$('zone').innerHTML='<b>'+rackName(zone)+' · schematic average</b><br>Outlet '+format(r.outlet_temperature)+' °C · supply '+format(r.proposed_supply)+' °C<br>IT '+format(r.P_IT)+' kW · facility '+format(r.P_DC)+' kW<br>Cooling-only power unmetered';
 }
 if(compare)content='<div class="group" id="peer"><h4>'+escapeHtml(data.other)+' · matched step</h4>'+(peer?'Outlet '+format(peer.outlet_temperature)+' °C · supply '+format(peer.proposed_supply)+' °C · '+format(peer.P_DC)+' kW<br>'+escapeHtml(peer.decision_reason):'Comparison episode ended')+'</div>'+content;
 $('info').innerHTML=content;
 $('info').querySelectorAll('details').forEach(el=>el.open=oldOpen.has(el.querySelector('summary').textContent));
 if($('toLoop'))$('toLoop').onclick=()=>setMode('LOOP');
}
function setMode(value){
 mode=value; const isLoop=mode==='LOOP';
 $('room').hidden=isLoop;$('loop').hidden=!isLoop;$('markers').hidden=isLoop;$('zone').hidden=true;
 $('caption').textContent=isLoop?'Schematic cooling circuit · replay readings and observed AAU outlet distribution':'Schematic - AAU data are averages, not per-rack measurements. Seeded rack colours are cosmetic.';
 document.querySelectorAll('.mode button').forEach(b=>b.setAttribute('aria-pressed',b.textContent===mode));
 room?.mode(mode);setLegend();update();
}
function pause(){$('replay').checked=false;clearInterval(timer);timer=null;$('play').textContent='▶';$('play').setAttribute('aria-label','Play replay');}
function play(){pause();$('replay').checked=true;$('play').textContent='Ⅱ';$('play').setAttribute('aria-label','Pause replay');timer=setInterval(()=>{
 $('step').value=(+$('step').value+1)%frames.length;update();},600/parseFloat($('speed').value));}
$('step').max=frames.length-1;$('end').textContent='Step '+data.horizon;$('step').oninput=()=>{pause();update();};
$('compare').disabled=!data.comparison.length;$('compare').onchange=update;
$('play').onclick=()=>timer?pause():play();$('speed').onchange=()=>{if(timer)play();};
$('replay').onchange=()=>{if($('replay').checked)play();else pause();};
$('home').onclick=()=>room?.home();$('legendToggle').onclick=()=>$('keys').hidden=!$('keys').hidden;
const main=document.querySelector('main');
function togglePanel(){main.classList.toggle(document.body.classList.contains('narrow')?'open':'closed');}
$('inspect').onclick=$('close').onclick=togglePanel;
new ResizeObserver(()=>document.body.classList.toggle('narrow',innerWidth<1100)).observe(document.body);
document.querySelectorAll('.mode button').forEach(btn=>btn.onclick=()=>setMode(btn.textContent));
setMode('3D');play();
