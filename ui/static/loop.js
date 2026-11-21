function makeLoop(host,data){
  // All geometry uses one floor projection so pipe endpoints meet the equipment.
  const point=(x,z,y=0)=>[486+(x-z)*28,193+(x+z)*13-y*23];
  const pts=vertices=>vertices.map(p=>p.join(',')).join(' ');
  function block(x,z,w,d,h,top,side='#909bad',front='#647286',heat=false){
    const a=point(x,z,h),b=point(x+w,z,h),c=point(x+w,z+d,h),e=point(x,z+d,h);
    return '<polygon fill="'+side+'" points="'+pts([e,c,point(x+w,z+d),point(x,z+d)])+'"/><polygon fill="'+front+'" points="'+pts([b,c,point(x+w,z+d),point(x+w,z)])+'"/><polygon '+(heat?'class="hallTop" ':'')+'fill="'+top+'" points="'+pts([a,b,c,e])+'"/>';
  }
  function pipe(points,color){return '<polyline points="'+pts(points.map(p=>point(...p)))+'" fill="none" stroke="'+color+'" stroke-width="9" stroke-linecap="round" stroke-linejoin="round"/>';}
  let grid='';
  for(let i=-14;i<=24;i++)for(const ends of [[[i,-14],[i,24]],[[-14,i],[24,i]]])grid+='<polyline points="'+pts(ends.map(p=>point(...p)))+'" fill="none" stroke="#30394b" stroke-width=".7"/>';
  let equipment=pipe([[-2,1],[-2,12],[9.95,12]],'#ef4a5c')+pipe([[3,1],[14,1],[14,8.6],[13.05,8.6]],'#5b9bff');
  for(let i=0;i<4;i++){
    equipment+=block(-3+i*1.5,-1,1.4,2,1.4,'#d6dfed');
    const [x,y]=point(-2.3+i*1.5,0,1.41);equipment+='<ellipse cx="'+x+'" cy="'+y+'" rx="10" ry="6" fill="#8393ab"/>';
  }
  for(const [x,z] of [[5,1],[14,5]])equipment+=block(x,z,.9,.9,.8,'#ccd8e6');
  const max=Math.max(...data.histogram,1);
  data.histogram.forEach((n,i)=>{equipment+='<g><title>'+format(data.edges[i],1)+'–'+format(data.edges[i+1],1)+' °C: '+n+' rows</title>'+block(3.6+i*.37,4.3,.31,1.25,2.8*n/max,'#83bbff','#4177b5','#5b9bff')+'</g>';});
  for(let i=0;i<3;i++)equipment+=block(10,8+i*1.7,3,1.55,1.9,'#facc15','#a8b1c1','#798497',true);
  const labels=[
    ['Cooling supply','proposed_supply',785,125,[9,1],'#78adff'],
    ['Cooling return','Cooling Return Average',150,243,[-2,9],'#ff7382'],
    ['Rack inlet','Inlet Temp Avg',806,305,[14,6],'#55ceff'],
    ['Rack outlet','outlet_temperature',725,550,[12,11],'#ffab62']
  ];
  const sensors=labels.map(([label,key,x,y,anchor,color],i)=>{
    const [ax,ay]=point(...anchor);return '<path d="M'+x+' '+(y+39)+' L'+ax+' '+ay+'" stroke="#b3c2d9" fill="none"/><circle cx="'+ax+'" cy="'+ay+'" r="5" fill="'+color+'"/><text x="'+x+'" y="'+y+'" fill="'+color+'" font-size="28" font-weight="700" id="sensor'+i+'"></text><text x="'+x+'" y="'+(y+34)+'" fill="#d3deef" font-size="16">'+label+'</text>';
  }).join('');
  host.innerHTML='<svg viewBox="0 0 980 660" preserveAspectRatio="xMidYMid meet" aria-label="Schematic cooling loop"><g>'+grid+equipment+'</g><g text-anchor="middle" font-size="16">'+sensors+'<text x="380" y="98" fill="#e8eefb">Cooling units</text><text x="265" y="456" fill="#c9d9f2">Outlet distribution</text><text x="265" y="484" font-size="14" fill="#c9d9f2">'+data.histogram.reduce((a,b)=>a+b,0)+' AAU samples</text><text x="455" y="580" fill="#e8eefb">Rack hall · '+escapeHtml(data.config)+'</text></g></svg>';
  return (row,ramp)=>{labels.forEach(([,key],i)=>document.getElementById('sensor'+i).textContent=format(row[key],1)+' °C');
    host.querySelectorAll('.hallTop').forEach(el=>el.setAttribute('fill',ramp(row.outlet_temperature)));};
}
