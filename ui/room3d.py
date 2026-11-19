"""A clearly labelled schematic three.js server-room replay."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit.components.v1 as components


def render_room(trajectory: pd.DataFrame, configuration: str) -> None:
    """Render two rows of cosmetic racks coloured from one AAU average trajectory."""
    records = trajectory[["step", "outlet_temperature", "action", "proposed_supply", "decision_reason"]].to_dict("records")
    payload = json.dumps(records)
    configuration_payload = json.dumps(configuration)
    components.html(
        f"""
        <div style="background:#07111f;color:#e2e8f0;font-family:system-ui;padding:8px">
          <strong>Schematic server room</strong><span style="color:#94a3b8"> — Schematic. AAU data are averages, not per-rack measurements.</span>
          <div style="display:flex;gap:12px;align-items:center;margin:8px 0">
            <button id="play">Play</button><input id="step" type="range" min="0" value="0" style="flex:1" />
            <select id="speed" aria-label="Playback speed"><option value="600">1×</option><option value="300">2×</option><option value="1200">½×</option></select><span id="label"></span>
          </div><small>24 °C 🔵 · 27.5 °C 🟢 · 31 °C 🔴 | Cosmetic rack variation</small><div id="decision" aria-live="polite"></div><div id="room"></div>
        </div>
        <script>{(Path(__file__).parent / "static/three.min.js").read_text(encoding="utf-8")}</script>
        <script>
        const data = {payload}; const configuration = {configuration_payload};
        const root = document.getElementById('room'); const slider = document.getElementById('step');
        const label = document.getElementById('label'); slider.max = Math.max(0, data.length - 1);
        try {{
        const scene = new THREE.Scene(); scene.background = new THREE.Color('#07111f');
        const camera = new THREE.PerspectiveCamera(48, 2.7, 0.1, 100); camera.position.set(7, 7, 10); camera.lookAt(0, 0, 0);
        const renderer = new THREE.WebGLRenderer({{antialias:true}}); renderer.setSize(root.clientWidth, 420); root.appendChild(renderer.domElement);
        scene.add(new THREE.AmbientLight(0xffffff, 1.1)); const light = new THREE.DirectionalLight(0x38bdf8, 1.4); light.position.set(5,8,5); scene.add(light);
        const floor = new THREE.Mesh(new THREE.PlaneGeometry(13, 7), new THREE.MeshStandardMaterial({{color:0x0f1d2f}})); floor.rotation.x=-Math.PI/2; scene.add(floor);
        scene.add(new THREE.GridHelper(14,28,0x38bdf8,0x26344e)); floor.position.y=-.02;
        function panel(w,h,d,x,y,z,color){{const mesh=new THREE.Mesh(new THREE.BoxGeometry(w,h,d),new THREE.MeshStandardMaterial({{color,transparent:true,opacity:.55}})); mesh.position.set(x,y,z);scene.add(mesh);}}
        panel(10,.04,1.6,0,.02,0,0x197694);
        if(configuration.includes('Plate')) panel(10,.12,2.4,0,2.3,0,0x7da6b8);
        if(configuration.includes('Wall')) panel(10,2.5,.12,0,1.25,-2,0x657892);
        const racks = []; const rackCount = configuration === 'PlateWall' ? 12 : configuration === 'Wall' ? 10 : 8;
        for (let row=0; row<2; row++) for (let i=0; i<rackCount; i++) {{
          const material = new THREE.MeshStandardMaterial({{color:0x38bdf8, emissive:0x001018}});
          const rack = new THREE.Mesh(new THREE.BoxGeometry(.58, 2.1, .8), material); rack.position.set((i-(rackCount-1)/2)*.75,1.05,row ? 1.4 : -1.4); scene.add(rack); racks.push(rack); for(let slot=0;slot<7;slot++) panel(.36,.028,.024,rack.position.x,.35+slot*.24,rack.position.z+.412,0x90d9e9);
          const plume = new THREE.Mesh(new THREE.ConeGeometry(.17,.9,10), new THREE.MeshBasicMaterial({{color:0x38bdf8,transparent:true,opacity:.42}})); plume.position.set(rack.position.x,2.65,rack.position.z); scene.add(plume); rack.userData.plume=plume;
        }}
        function update() {{ const current=data[Number(slider.value)] || {{step:0,outlet_temperature:0}}; const t=current.outlet_temperature; const ratio=Math.max(0,Math.min(1,(t-24)/7));
          racks.forEach((rack,index)=>{{ const shifted=Math.max(0,Math.min(1,ratio+((index%5)-2)*.025)); const color=new THREE.Color().setHSL((1-shifted)*.62,.86,.46); rack.material.color=color; rack.material.emissive=color.clone().multiplyScalar(.16); rack.userData.plume.material.color=color; }});
          document.getElementById('decision').textContent=current.decision_reason;
          label.textContent=`step ${{current.step}} · outlet ${{t.toFixed(1)}} °C · supply ${{current.proposed_supply.toFixed(1)}} °C · ${{["Decrease","Maintain","Increase"][current.action]}}`; renderer.render(scene,camera); }}
        new ResizeObserver(()=>{{renderer.setSize(root.clientWidth,420);camera.aspect=root.clientWidth/420;camera.updateProjectionMatrix();update()}}).observe(root);
        let angle=.6; renderer.domElement.onpointermove=event=>{{if(event.buttons===1){{angle+=event.movementX*.01;camera.position.set(12*Math.sin(angle),7,12*Math.cos(angle));camera.lookAt(0,0,0);update()}}}};
        slider.oninput=update; let timer=null; const play=document.getElementById('play');play.onclick=()=>{{if(timer){{clearInterval(timer);timer=null;play.textContent='Play';return}} play.textContent='Pause';timer=setInterval(()=>{{slider.value=(Number(slider.value)+1)%data.length;update()}},Number(document.getElementById('speed').value))}};update();
        }} catch(error) {{root.textContent='3D view unavailable in this browser. Read outlet, supply and action in the episode replay.'}}
        </script>
        """,
        height=600,
    )
