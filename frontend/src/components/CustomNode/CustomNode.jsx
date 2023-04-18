import { Handle, Position, } from 'reactflow';
import './customNode.css'
import { useContext, useState } from 'react';
import { MainContext } from '../context/MainContext';

const handleStyle = { left: 10 };

function CustomNode({ id, data, isConnectable }) {
  const { setNodes, setEdges, nodes, edges } = useContext(MainContext)
  const [ buttonActive, setButtonActive ] = useState(false) 

  const handleClick = () => {
    console.log("CLICK")

    if (!buttonActive) {
      setButtonActive(true)

      let nodesGroup = JSON.parse(JSON.stringify(data.include.nodes))
      let egdesGgroup =  JSON.parse(JSON.stringify(data.include.edges))

      const position = nodes.filter(el => el.id === id)[0].position  // TODO: get position from props

      const groupEdge = {
        id: "edge_" + id + "_open",
        source: id,
        target:  'group' + id + '_open',
        animated: true 
      }
  
      let xMas = []
      let yMas = []

      for (let i=0; i < nodesGroup.length; i++) {
        xMas.push(nodesGroup[i].position.x)
        yMas.push(nodesGroup[i].position.y)
  
        nodesGroup[i].parentNode =  'group' + id + '_open'
        nodesGroup[i].extent = 'parent'
        nodesGroup[i].id += id + '_open'
      }
  
      let xMax = xMas.indexOf(Math.max(...xMas));
      let xMin = xMas.indexOf(Math.min(...xMas));
      let yMax = yMas.indexOf(Math.max(...yMas));
      let yMin = yMas.indexOf(Math.min(...yMas));
      let temp = nodesGroup[0].position.x - nodesGroup[xMin].position.x
  
      const groupNode = {
        id:  'group' + id + '_open',
        position: {x: position.x, y: position.y+80},
        data: {
          label: 'group_'+data.label
        },
        style: { backgroundColor: 'rgba(255, 0, 0, 0.2)', 
        width: (nodesGroup[xMax].position.x + 150) - nodesGroup[xMin].position.x + 100, 
        height: (nodesGroup[yMax].position.y + 40) - nodesGroup[yMin].position.y + 100,
        }
      }

      
      for (let i=1; i < nodesGroup.length; i++) {
        nodesGroup[i].position = {
          x: 50 + temp + nodesGroup[i].position.x - nodesGroup[0].position.x,
          y: 50 + nodesGroup[i].position.y - nodesGroup[0].position.y
        }
      }

      nodesGroup[0].position = {
        x: temp + 50,
        y: 50,
      }

      for (let i=0; i < egdesGgroup.length; i++) {
        egdesGgroup[i].id += id +'_open'
        egdesGgroup[i].source += id + '_open'
        egdesGgroup[i].target += id + '_open'
      }
      
      console.log("GROUP_EDGES", egdesGgroup)

      setNodes((nds) => nds.concat(groupNode))
      setEdges((edg) => edg.concat(groupEdge))
      setNodes((nds) => nds.concat(...nodesGroup))
      setEdges((edg) => edg.concat(...egdesGgroup))

    } else {
      setButtonActive(false)
      setEdges(edges.filter(edg => edg.id.slice(-(id.length + 5)) != id + '_open'))
      setNodes(nodes.filter(nds => nds.id.slice(-(id.length + 5)) != id + '_open'))
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
