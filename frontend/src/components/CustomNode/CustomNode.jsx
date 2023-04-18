import { Handle, Position, } from 'reactflow';
import './customNode.css'
import { useContext, useState } from 'react';
import { MainContext } from '../context/MainContext';

const handleStyle = { left: 10 };

function CustomNode({ id, data, isConnectable }) {
  const { setNodes, setEdges, nodes, edges } = useContext(MainContext)
  const [ buttonActive, setButtonActive ] = useState(false) 

  const handleClick = () => {
    if (!buttonActive) {
      setButtonActive(true)

      const position = nodes.filter(el => el.id === id)[0].position  // TODO: get position from props

      const groupEdge = {
        id: "edge_" + id + "_open",
        source: id,
        target: id + '_open',
        animated: true 
      }
  
      let xMas = []
      let yMas = []
  
      for (let i=0; i < data.include.nodes.length; i++) {
        xMas.push(data.include.nodes[i].position.x)
        yMas.push(data.include.nodes[i].position.y)
  
        data.include.nodes[i].parentNode = id + '_open'
        data.include.nodes[i].extent = 'parent'
        data.include.nodes[i].id += id + '_open'
      }
  
      let xMax = xMas.indexOf(Math.max(...xMas));
      let xMin = xMas.indexOf(Math.min(...xMas));
      let yMax = yMas.indexOf(Math.max(...yMas));
      let yMin = yMas.indexOf(Math.min(...yMas));
  
      let temp = data.include.nodes[0].position.x - data.include.nodes[xMin].position.x
  
      const groupNode = {
        id: id + '_open',
        position: {x: position.x, y: position.y+80},
        data: {
          label: 'group_'+data.label
        },
        style: { backgroundColor: 'rgba(255, 0, 0, 0.2)', 
        width: (data.include.nodes[xMax].position.x + 150) - data.include.nodes[xMin].position.x + 100, 
        height: (data.include.nodes[yMax].position.y + 40) - data.include.nodes[yMin].position.y + 100,
        }
      }
  
      for (let i=1; i < data.include.nodes.length; i++) {
        data.include.nodes[i].position = {
          x: 50 + temp + data.include.nodes[i].position.x - data.include.nodes[0].position.x,
          y: 50 + data.include.nodes[i].position.y - data.include.nodes[0].position.y
        }
      }
  
      data.include.nodes[0].position = {
        x: temp + 50,
        y: 50,
      }

  
      for (let i=0; i < data.include.edges.length; i++) {
        data.include.edges[i].id += id +'_open'
        data.include.edges[i].source += '_open'
        data.include.edges[i].target += '_open'
      }
      
      console.log("GROUP_EDGES", data.include.edges)

      setNodes((nds) => nds.concat(groupNode))
      setEdges((edg) => edg.concat(groupEdge))
      setNodes((nds) => nds.concat(...data.include.nodes))
      setEdges((edg) => edg.concat(...data.include.edges))

    } else {
      setButtonActive(false)
      setEdges(edges.filter(edg => edg.id.slice(-50) != id +'_open'))
      setNodes(nodes.filter(nds => nds.id.slice(-50) != id +'_open'))
    }
  };


  return (
    <div className="custom-node">
      <Handle type="target" position={Position.Top} isConnectable={isConnectable} />
      <div>
        <label htmlFor="text">{data.label}</label>
        <button className='open-button' onClick={() => handleClick()}>Open</button>
      </div>
      <Handle
        type="source"
        position={Position.Bottom}
        id="a"
        style={handleStyle}
        isConnectable={isConnectable}
      />
      <Handle type="source" position={Position.Bottom} id="b" isConnectable={isConnectable} />
    </div>
  );
}

export default CustomNode;
