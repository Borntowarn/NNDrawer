import NodeUpdate from '../../NodeUpdate/NodeUpdate'
import './ParamsArea.css'

export default function ParamsArea() {
  return (
    <div className='params-area'>
      <input placeholder='Parameter title' className='nodes-search'/>
      <NodeUpdate />
    </div>
  )
}
