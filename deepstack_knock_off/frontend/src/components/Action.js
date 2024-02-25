

const Action = ({ action, onClick, isSelected }) => {

  const className = `action-item ${isSelected ? 'selected' : ''}`;

  return (
    <li className={className} onClick={onClick}>
      {action}
    </li>
  );
};

export default Action;