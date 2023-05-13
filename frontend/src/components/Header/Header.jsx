import './Header.css'

export default function Header() {
  const handleClick = () => {
    
  } 

  return (
    <div className='header-area'>
        <div className='header-title'>
            NNdrawer
        </div>
        <a onClick={() => handleClick()} className='user-logo'>
            AMOGUS
        </a>
    </div>
  )
}
