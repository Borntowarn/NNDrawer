import NodesArea from '../areas/NodesArea/NodesArea';
import ParamsArea from '../areas/ParamsArea/ParamsArea';
import './SideBar.css'

export default () => {
  return (
    <aside className='side-area'>
      <NodesArea />
      <ParamsArea />
    </aside>
  );
};