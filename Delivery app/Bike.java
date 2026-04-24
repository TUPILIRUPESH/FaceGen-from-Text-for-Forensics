class Bike extends Vehicle {
    @Override
    public void assign() {
        System.out.println("the delivery has been assigned in bike");
    }

    @Override
    public void dispatch() {
        System.out.println("the delivery has been dispatched in bike");
    }
}