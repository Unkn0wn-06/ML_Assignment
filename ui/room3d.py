"""A clearly labelled schematic three.js server-room replay."""

from __future__ import annotations

import json

import pandas as pd
import streamlit.components.v1 as components


def render_room(trajectory: pd.DataFrame, configuration: str) -> None:
    """Render two rows of cosmetic racks coloured from one AAU average trajectory."""
    records = trajectory[["step", "outlet_temperature"]].to_dict("records")
    payload = json.dumps(records)
    configuration_payload = json.dumps(configuration)
    components.html(
        f"""
        <div style="background:#07111f;color:#e2e8f0;font-family:system-ui;padding:8px">
          <strong>Schematic server room</strong><span style="color:#94a3b8"> — AAU data are averages, not per-rack measurements.</span>
          <div style="display:flex;gap:12px;align-items:center;margin:8px 0">
            <button id="play">Play</button><input id="step" type="range" min="0" value="0" style="flex:1" />
            <span id="label"></span>
          </div><div id="room"></div>
        </div>
        <script src="https://cdn.jsdelivr.net/npm/three@0.161.0/build/three.min.js"></script>
        <script>
        const data = {payload}; const configuration = {configuration_payload};
        const root = document.getElementById('room'); const slider = document.getElementById('step');
        const label = document.getElementById('label'); slider.max = Math.max(0, data.length - 1);
        const scene = new THREE.Scene(); scene.background = new THREE.Color('#07111f');
        const camera = new THREE.PerspectiveCamera(48, 2.7, 0.1, 100); camera.position.set(7, 7, 10); camera.lookAt(0, 0, 0);
        const renderer = new THREE.WebGLRenderer({{antialias:true}}); renderer.setSize(760, 280); root.appendChild(renderer.domElement);
        scene.add(new THREE.AmbientLight(0xffffff, 1.1)); const light = new THREE.DirectionalLight(0x38bdf8, 1.4); light.position.set(5,8,5); scene.add(light);
        const floor = new THREE.Mesh(new THREE.PlaneGeometry(13, 7), new THREE.MeshStandardMaterial({{color:0x0f1d2f}})); floor.rotation.x=-Math.PI/2; scene.add(floor);
        const racks = []; const rackCount = configuration === 'PlateWall' ? 12 : configuration === 'Wall' ? 10 : 8;
        for (let row=0; row<2; row++) for (let i=0; i<rackCount; i++) {{
          const material = new THREE.MeshStandardMaterial({{color:0x38bdf8, emissive:0x001018}});
          const rack = new THREE.Mesh(new THREE.BoxGeometry(.58, 2.1, .8), material); rack.position.set((i-(rackCount-1)/2)*.75,1.05,row ? 1.4 : -1.4); scene.add(rack); racks.push(rack);
          const plume = new THREE.Mesh(new THREE.ConeGeometry(.17,.9,10), new THREE.MeshBasicMaterial({{color:0x38bdf8,transparent:true,opacity:.42}})); plume.position.set(rack.position.x,2.65,rack.position.z); scene.add(plume); rack.userData.plume=plume;
        }}
        function update() {{ const current=data[Number(slider.value)] || {{step:0,outlet_temperature:0}}; const t=current.outlet_temperature; const ratio=Math.max(0,Math.min(1,(t-20)/12));
          racks.forEach((rack,index)=>{{ const shifted=Math.max(0,Math.min(1,ratio+((index%5)-2)*.025)); const color=new THREE.Color().setHSL((1-shifted)*.34,.86,.46); rack.material.color=color; rack.material.emissive=color.clone().multiplyScalar(.16); rack.userData.plume.material.color=color; }});
          label.textContent=`step ${{current.step}} · avg outlet ${{t.toFixed(2)}} °C`; renderer.render(scene,camera); }}
        slider.addEventListener('input',update); let timer; document.getElementById('play').onclick=()=>{{clearInterval(timer); timer=setInterval(()=>{{slider.value=(Number(slider.value)+1)%(Number(slider.max)+1);update()}},550)}}; update();
        </script>
        """,
        height=360,
    )
