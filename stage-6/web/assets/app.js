const byId = id => document.getElementById(id);
const map = L.map('map', {preferCanvas:true, zoomSnap:0.25, minZoom:0}).setView([20,0], 1);
const worldBounds = L.latLngBounds([-78,-180],[83,180]);
const showWorld = () => map.fitBounds(worldBounds,{padding:[14,14],animate:false});
let tripBounds;
byId('world-view').onclick=showWorld;
byId('trip-view').onclick=()=>map.fitBounds(tripBounds,{padding:[24,24],maxZoom:17,animate:false});
showWorld();
fetch('/assets/world.geojson').then(response=>{
  if (!response.ok) throw Error('World map unavailable');
  return response.json();
}).then(world=>{
  L.geoJSON(world,{interactive:false,attribution:'Made with <a href="https://www.naturalearthdata.com/">Natural Earth</a>',
    style:{color:'#9db4b0',weight:1,fillColor:'#e9eee3',fillOpacity:1}}).addTo(map);
}).catch(error=>console.error(error));
for (const [name,level] of [['roadPane',410],['routePane',420],['gpsPane',430]]) {
  const pane=map.createPane(name);
  pane.style.zIndex=level;
  pane.style.pointerEvents='none';
}
const layers = {};
const number = n => n.toLocaleString();
const percent = n => n == null ? 'N/A' : (100*n).toFixed(2)+'%';
const score = n => n == null ? 'N/A' : n.toFixed(4);
function clearResult() {
  byId('trip-view').disabled=true;
  for (const group of Object.values(layers)) map.removeLayer(group);
  for (const key of ['accuracy','continuity','sequence','macro']) byId(key).textContent='—';
  byId('receipt').hidden=true;
  byId('coverage').textContent='Select a trip and run reconstruction.';
  byId('map-title').textContent='Trip map';
  byId('timing').textContent='';
}
function display(d) {
  const m=d.metrics, g=d.map;
  layers.roads=L.polyline(g.roads,{pane:'roadPane',color:'#acb7af',weight:1.1,interactive:false});
  layers.route=L.polyline(g.runs,{pane:'routePane',color:'#14634e',weight:5,opacity:.94,interactive:false});
  layers.gps=L.layerGroup(g.gps.map(p=>L.circleMarker(p,{pane:'gpsPane',radius:1.7,stroke:false,fillColor:'#bc4f24',fillOpacity:.68,interactive:false})));
  for (const [key,group] of Object.entries(layers)) if (byId(key).checked) group.addTo(map);
  tripBounds=L.latLngBounds(g.gps);
  for (const run of g.runs) for (const point of run) tripBounds.extend(point);
  byId('trip-view').disabled=false;
  byId('trip-view').click();
  byId('map-title').textContent='Trip '+d.trip;
  byId('timing').textContent='Inference · '+number(d.inference_ms)+' ms';
  byId('accuracy').textContent=percent(m.accuracy);
  byId('continuity').textContent=percent(m.continuity);
  byId('sequence').textContent=score(m.sequence_f1);
  byId('macro').textContent=score(m.macro_f1);
  byId('coverage').textContent=number(m.correct)+' correct out of '+number(m.eligible)+' scorable GPS points. '+number(m.observations)+' observations in this trip.';
  byId('request-id').textContent=d.request_id;
  byId('receipt').hidden=false;
}
for (const key of ['roads','route','gps']) byId(key).onchange=()=>{
  if (!layers[key]) return;
  if (byId(key).checked) layers[key].addTo(map); else map.removeLayer(layers[key]);
};
byId('trip').onchange=()=>{clearResult();for (const k of Object.keys(layers)) delete layers[k];byId('status').textContent='';};
byId('trip-form').onsubmit=async e=>{
  e.preventDefault();clearResult();
  for (const k of Object.keys(layers)) delete layers[k];
  byId('trip').disabled=byId('run').disabled=true;
  byId('status').className='';byId('status').textContent='Reconstructing route…';
  try {
    const r=await fetch('/api/trips/'+encodeURIComponent(byId('trip').value)+'/predict',{method:'POST'});
    if (!r.ok) throw Error((await r.json()).detail || 'Request failed.');
    display(await r.json());byId('status').textContent='Route received from the API.';
  } catch(error){byId('status').className='error';byId('status').textContent=error.message;}
  finally{byId('trip').disabled=byId('run').disabled=false;}
};
(async()=>{
  try{
    const r=await fetch('/api/trips');if(!r.ok)throw Error();
    const d=await r.json();
    for (const trip of d.trips) byId('trip').add(new Option(trip,trip));
    byId('partition').textContent=(d.mode==='student-test'?'Student-test rehearsal':'Evaluator partition')+' · '+d.trips.length+' trips';
    byId('connection').textContent='API connected';
    byId('trip').disabled=byId('run').disabled=false;
  } catch {byId('connection').textContent='API unavailable';byId('status').textContent='Reconnect to Laptop 1 and reload this page.';}
})();
