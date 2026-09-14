export class ChartRenderer {
  constructor(canvas){this.canvas=canvas;this.ctx=canvas.getContext('2d');}
  draw(metrics){
    const r=this.canvas.getBoundingClientRect(),d=Math.min(2,devicePixelRatio||1),w=r.width,h=r.height;
    if(this.canvas.width!==Math.round(w*d)||this.canvas.height!==Math.round(h*d)){this.canvas.width=Math.round(w*d);this.canvas.height=Math.round(h*d);}
    const c=this.ctx;c.setTransform(d,0,0,d,0,0);c.clearRect(0,0,w,h);c.fillStyle='#071018';c.fillRect(0,0,w,h);
    if(metrics.length<2)return;
    const vals=metrics.map(m=>m.meanReturn);let lo=Math.min(...vals),hi=Math.max(...vals);if(lo===hi){lo-=1;hi+=1;}
    for(let i=1;i<metrics.length;i++){
      if(metrics[i].curriculum!==metrics[i-1].curriculum){
        const x=(i/(metrics.length-1))*w;c.strokeStyle='rgba(245,189,92,.44)';c.lineWidth=1;c.setLineDash([3,3]);c.beginPath();c.moveTo(x,0);c.lineTo(x,h);c.stroke();c.setLineDash([]);
      }
      if(metrics[i].validation){
        const x=(i/(metrics.length-1))*w;c.fillStyle=metrics[i].validation.regression?'rgba(239,98,152,.85)':'rgba(104,226,193,.82)';c.beginPath();c.arc(x,6,2.5,0,Math.PI*2);c.fill();
      }
    }
    c.strokeStyle='rgba(96,220,240,.8)';c.lineWidth=1.5;c.beginPath();
    for(let i=0;i<vals.length;i++){const x=(i/(vals.length-1))*w,y=h-8-((vals[i]-lo)/(hi-lo))*(h-18);if(i===0)c.moveTo(x,y);else c.lineTo(x,y);}c.stroke();
    c.font='10px system-ui';c.fillStyle='rgba(190,220,230,.58)';c.fillText(`return ${vals.at(-1).toFixed(2)}`,8,13);
  }
}
