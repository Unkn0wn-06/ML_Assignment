function buildRoom(root, payload, selectZone) {
  const T=THREE, scene=new T.Scene(), renderer=new T.WebGLRenderer({antialias:true,alpha:false});
  renderer.setPixelRatio(Math.min(devicePixelRatio,1.5)); root.append(renderer.domElement);
  scene.background=new T.Color('#dfe4ea');
  const camera=new T.OrthographicCamera(-10,10,8,-8,.1,150);
  let yaw=.75, pitch=.52, zoom=1, plan=false, active=true;
  const cube=new T.BoxGeometry(), transform=new T.Object3D(), color=new T.Color();
  function box(w,h,d,x,y,z,tint,opacity=1){
    const mesh=new T.Mesh(cube,new T.MeshLambertMaterial({color:tint,transparent:opacity<1,opacity,depthWrite:opacity===1}));
    mesh.scale.set(w,h,d);mesh.position.set(x,y,z);scene.add(mesh);return mesh;
  }
  scene.add(new T.HemisphereLight(0xffffff,0xb7bdca,1.65));
  const sun=new T.DirectionalLight(0xffffff,1.6);sun.position.set(-4,10,6);scene.add(sun);
  box(49,.08,49,0,-.06,0,'#e8ebef');scene.add(new T.GridHelper(48,48,0xbcc4cf,0xbcc4cf));
  const count=40, racks=new T.InstancedMesh(cube,new T.ShaderMaterial({
    vertexShader:`varying vec3 tint;varying vec3 pos;varying vec3 norm;
    void main(){pos=position;norm=normal;tint=instanceColor;gl_Position=projectionMatrix*modelViewMatrix*instanceMatrix*vec4(position,1.);}`,
    fragmentShader:`varying vec3 tint;varying vec3 pos;varying vec3 norm;
    void main(){float y=pos.y+.5;vec3 c=mix(vec3(.045,.28,.42),tint,smoothstep(.05,.94,y));
    float grille=step(abs(pos.x),.39)*step(.04,y)*step(y,.88)*step(.72,fract(y*33.))*abs(norm.z);
    gl_FragColor=vec4(c*(.84+.16*norm.y-.12*norm.x)*(1.-.32*grille),1.);
    #include <colorspace_fragment>
    }`
  }),count);
  const anchors=[], offsets=[];let random=payload.seed>>>0;
  for(let i=0;i<count;i++){
    const row=Math.floor(i/10),x=(i%10-4.5)*.96,z=[-3,-1.35,1.35,3][row];
    transform.position.set(x,1.1,z);transform.scale.set(.89,2.2,.9);transform.updateMatrix();
    racks.setMatrixAt(i,transform.matrix);racks.setColorAt(i,color.set('#38bdf8'));
    random=(Math.imul(random,1664525)+1013904223)>>>0;offsets.push((random/4294967296-.5)*1.2);
    if([3,16,24,38].includes(i))anchors.push(new T.Vector3(x,2.5,z));
  }
  scene.add(racks);
  for(const z of [-2.8,2.8]){
    box(11,.1,.12,0,2.8,z,'#5b9bff');box(11,.1,.12,0,2.8,z+.23,'#ef4a5c');
    const unit=box(1,2.7,1.3,5.7,1.35,z,'#24344a');
    if(z<0){const edges=new T.LineSegments(new T.EdgesGeometry(cube),new T.LineBasicMaterial({color:0xfacc15}));unit.add(edges);}
  }
  if(payload.config.includes('Plate'))box(9.8,.08,1.8,0,2.48,0,'#8bb5c5',.24);
  if(payload.config.includes('Wall'))box(.08,2.45,2.3,-4.96,1.23,0,'#86a3b2',.45);
  const markers=document.getElementById('markers'), pins=anchors.map((p,i)=>{
    const btn=document.createElement('button');btn.className='pin';btn.title='Inspect schematic zone '+(i+1);
    btn.onclick=()=>selectZone(i,true);btn.onmouseenter=()=>selectZone(i,false);btn.onmouseleave=()=>selectZone(null,false);
    markers.append(btn);return btn;
  });
  function draw(){
    if(!active)return;
    const w=root.clientWidth,h=root.clientHeight;if(!w||!h)return;
    const half=Math.max(4.4,6.6*h/w)/zoom;
    camera.left=-half*w/h;camera.right=-camera.left;camera.top=half;camera.bottom=-half;
    camera.position.set(plan?0:21*Math.cos(pitch)*Math.sin(yaw),plan?22:21*Math.sin(pitch),plan?.001:21*Math.cos(pitch)*Math.cos(yaw));
    camera.lookAt(0,1,0);camera.updateProjectionMatrix();camera.updateMatrixWorld();renderer.render(scene,camera);
    anchors.forEach((pos,i)=>{const p=pos.clone().project(camera);pins[i].style.left=((p.x+1)*50)+'%';pins[i].style.top=((1-p.y)*50)+'%';});
  }
  const canvas=renderer.domElement;
  canvas.onpointerdown=e=>canvas.setPointerCapture(e.pointerId);
  canvas.onpointermove=e=>{if(e.buttons===1&&!plan){yaw-=e.movementX*.008;pitch=Math.max(.15,Math.min(1.4,pitch+e.movementY*.007));draw();}};
  canvas.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(.55,Math.min(2.8,zoom*Math.exp(-e.deltaY*.001)));draw();},{passive:false});
  new ResizeObserver(()=>{renderer.setSize(root.clientWidth,root.clientHeight,false);draw();}).observe(root);
  return {mode(value){plan=value==='PLAN';active=value!=='LOOP';draw();},home(){yaw=.75;pitch=.52;zoom=1;draw();},
    update(row,ramp,status){for(let i=0;i<count;i++)racks.setColorAt(i,color.set(ramp(row.outlet_temperature+offsets[i])));
      racks.instanceColor.needsUpdate=true;pins.forEach(btn=>{btn.textContent=status.icon;btn.style.background=status.color;btn.setAttribute('aria-label',btn.title+' · '+status.label);});draw();}};
}
